from .layers.transformer import *
from .layers.improved_transformer import *
from .layers.positional_encoding import *
from .model_utils import _make_seq_first, _make_batch_first, \
    _get_padding_mask, _get_key_padding_mask, _get_group_mask


class PositionalEncodingRFF(nn.Module):
    def __init__(self, d_model, d_input=2, std_dev=2.0):
        super(PositionalEncodingRFF, self).__init__()
        frequency_matrix = torch.normal(mean=torch.zeros(d_input, d_model//2),
                                        std=std_dev)        
        self.register_buffer('frequency_matrix', frequency_matrix)

    def forward(self, coordinates):
        """Creates Fourier features from coordinates.

        Args:
            coordinates (torch.Tensor): Shape (num_points, coordinate_dim)
        """
        # The coordinates variable contains a batch of vectors of dimension
        # coordinate_dim. We want to perform a matrix multiply of each of these
        # vectors with the frequency matrix. I.e. given coordinates of
        # shape (num_points, coordinate_dim) we perform a matrix multiply by
        # the transposed frequency matrix of shape (coordinate_dim, num_frequencies)
        # to obtain an output of shape (num_points, num_frequencies).
        prefeatures = torch.matmul(coordinates, self.frequency_matrix)
        # Calculate cosine and sine features
        cos_features = torch.cos(2 * math.pi * prefeatures)
        sin_features = torch.sin(2 * math.pi * prefeatures)
        # Concatenate sine and cosine features
        return torch.cat((cos_features, sin_features), dim=-1)  


class CADEmbedding(nn.Module):
    """Embedding: positional embed + command embed + parameter embed + group embed (optional)"""
    def __init__(self, cfg, seq_len, use_group=False, group_len=None, embed_type='quantize'):
        super().__init__()

        self.command_embed = nn.Embedding(cfg.n_commands, cfg.d_model)

        if embed_type == 'quantize':
            self.embed_type = 'quantize'
            args_dim = cfg.args_dim + 1
            self.arg_embed = nn.Embedding(args_dim, 64, padding_idx=0)
            self.embed_fcn = nn.Linear(64 * cfg.n_args, cfg.d_model)
        elif 'rff1d' in embed_type:
            self.embed_type = 'rff1d'
            rff_std = float(embed_type.split('rff1d')[1])
            self.arg_embed = PositionalEncodingRFF(64, d_input=1, std_dev=rff_std)
            self.embed_fcn = nn.Linear(64 * cfg.n_args, cfg.d_model)

        # use_group: additional embedding for each sketch-extrusion pair
        self.use_group = use_group
        if use_group:
            if group_len is None:
                group_len = cfg.max_num_groups
            self.group_embed = nn.Embedding(group_len + 2, cfg.d_model)

        self.pos_encoding = PositionalEncodingLUT(cfg.d_model, max_len=seq_len+2)

    def forward(self, commands, args, groups=None):
        S, N = commands.shape

        if self.embed_type == 'quantize':
            src = self.command_embed(commands.long()) + \
                self.embed_fcn(self.arg_embed((args + 1).long()).view(S, N, -1))  # shift due to -1 PAD_VAL
        elif 'rff1d' in self.embed_type:
            args_preprocessed = (args.reshape(-1, 1) + 1).float()
            src = self.command_embed(commands.long()) + \
                self.embed_fcn(self.arg_embed(args_preprocessed).view(S, N, -1))  # shift due to -1 PAD_VAL

        if self.use_group:
            src = src + self.group_embed(groups.long())

        src = self.pos_encoding(src)

        return src


class ConstEmbedding(nn.Module):
    """learned constant embedding"""
    def __init__(self, cfg, seq_len):
        super().__init__()

        self.d_model = cfg.d_model
        self.seq_len = seq_len

        self.PE = PositionalEncodingLUT(cfg.d_model, max_len=seq_len)

    def forward(self, z):
        N = z.size(1)
        src = self.PE(z.new_zeros(self.seq_len, N, self.d_model))
        return src


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        seq_len = cfg.max_total_len
        self.use_group = cfg.use_group_emb
        self.embedding = CADEmbedding(cfg, seq_len, use_group=self.use_group, embed_type=cfg.embed_type)

        encoder_layer = TransformerEncoderLayerImproved(cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        encoder_norm = LayerNorm(cfg.d_model)
        self.encoder = TransformerEncoder(encoder_layer, cfg.n_layers, encoder_norm)

    def forward(self, commands, args):
        padding_mask, key_padding_mask = _get_padding_mask(commands, seq_dim=0), _get_key_padding_mask(commands, seq_dim=0)
        group_mask = _get_group_mask(commands, seq_dim=0) if self.use_group else None

        src = self.embedding(commands, args, group_mask)

        memory = self.encoder(src, mask=None, src_key_padding_mask=key_padding_mask)

        z = (memory * padding_mask).sum(dim=0, keepdim=True) / padding_mask.sum(dim=0, keepdim=True) # (1, N, dim_z)
        return z


class FCN(nn.Module):
    def __init__(self, d_model, n_commands, n_args, args_dim=256, args_std=""):
        super().__init__()

        self.n_args = n_args
        self.args_dim = args_dim

        self.command_fcn = nn.Linear(d_model, n_commands)
        self.args_fcn = nn.Linear(d_model, n_args * args_dim)
        
        if len(args_std) > 0:
            if 'kaiming' in args_std:
                print(f'use kaiming init on pred with std={float(self.args_fcn.weight.std())}.')
                nn.init.kaiming_normal_(self.args_fcn.weight, mode="fan_in")
            else:
                nn.init.normal_(self.args_fcn.weight, std=float(args_std))
                print(f'use normal init on pred with std={float(self.args_fcn.weight.std())}.')

    def forward(self, out):
        S, N, _ = out.shape

        command_logits = self.command_fcn(out)  # Shape [S, N, n_commands]

        args_logits = self.args_fcn(out)  # Shape [S, N, n_args * args_dim]
        args_logits = args_logits.reshape(S, N, self.n_args, self.args_dim)  # Shape [S, N, n_args, args_dim]

        return command_logits, args_logits


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()

        self.embedding = ConstEmbedding(cfg, cfg.max_total_len)

        decoder_layer = TransformerDecoderLayerGlobalImproved(cfg.d_model, cfg.dim_z, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        decoder_norm = LayerNorm(cfg.d_model)
        self.decoder = TransformerDecoder(decoder_layer, cfg.n_layers_decode, decoder_norm)

        args_dim = cfg.args_dim + 1

        if cfg.pred_type == 'quantize':
            self.fcn = FCN(cfg.d_model, cfg.n_commands, cfg.n_args, args_dim)
        elif 'conv' in cfg.pred_type:
            self.fcn = FCN(cfg.d_model, cfg.n_commands, cfg.n_args, 1, cfg.pred_type.split('conv')[1])

    def forward(self, z):
        src = self.embedding(z)
        out = self.decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None)

        command_logits, args_logits = self.fcn(out)

        out_logits = (command_logits, args_logits)
        return out_logits


class Bottleneck(nn.Module):
    def __init__(self, cfg):
        super(Bottleneck, self).__init__()

        self.bottleneck = nn.Sequential(nn.Linear(cfg.d_model, cfg.dim_z),
                                        nn.Tanh())

    def forward(self, z):
        return self.bottleneck(z)


class CADTransformer(nn.Module):
    def __init__(self, cfg):
        super(CADTransformer, self).__init__()

        self.args_dim = cfg.args_dim + 1

        self.encoder = Encoder(cfg)

        self.bottleneck = Bottleneck(cfg)

        self.decoder = Decoder(cfg)
        self.cfg = cfg


    def forward(self, commands_enc, args_enc,
                z=None, return_tgt=True, encode_mode=False):
        commands_enc_, args_enc_ = _make_seq_first(commands_enc, args_enc)  # Possibly None, None

        if z is None:
            z = self.encoder(commands_enc_, args_enc_)
            z = self.bottleneck(z)
        else:
            z = _make_seq_first(z)

        if encode_mode: return _make_batch_first(z)

        out_logits = self.decoder(z)
        out_logits = _make_batch_first(*out_logits)

        res = {
            "command_logits": out_logits[0],
            "args_logits": out_logits[1]
        }

        if return_tgt:
            res["tgt_commands"] = commands_enc
            res["tgt_args"] = args_enc

        return res
