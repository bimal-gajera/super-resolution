"""AdcSR: Adversarial Diffusion Compression for Real-World Image Super-Resolution (Chen et al., CVPR 2025).

Ported from https://github.com/Guaishou74851/AdcSR (Apache-2.0, commit d0b2871): ``model.py`` (``Net``,
``halve_channels``), ``forward.py`` (slimmed block forwards) and the discriminator set-up of ``train.py``.
Needs diffusers/peft (``envs/adcsr``, ``requirements-adcsr.txt``); they are imported lazily so that the rest of
srbench works without them.
"""
import copy
import torch
import types
from os import path as osp
from torch import nn

from srbench.utils.registry import ARCH_REGISTRY

ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))


def resolve_path(path):
    """Resolve a relative path against the repo root when it exists there; anything else (absolute paths, Hugging
    Face repo ids such as ``Manojb/stable-diffusion-2-1-base``) is returned unchanged."""
    if path is None or osp.isabs(path):
        return path
    local = osp.join(ROOT, path)
    return local if osp.exists(local) else path


def load_sd_unet(sd_model, pretrained=True):
    """The Stable Diffusion 2.1-base U-Net (diffusers layout); ``pretrained=False`` only reads its config."""
    from diffusers import UNet2DConditionModel
    sd_model = resolve_path(sd_model)
    if pretrained:
        return UNet2DConditionModel.from_pretrained(sd_model, subfolder='unet')
    return UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(sd_model, subfolder='unet'))


def build_half_decoder(ckpt_path=None):
    """The channel-halved SD VAE decoder pretrained in AdcSR stage 1 (``halfDecoder.ckpt``)."""
    from diffusers.models.autoencoders.vae import Decoder
    decoder = Decoder(
        in_channels=4,
        out_channels=3,
        up_block_types=['UpDecoderBlock2D' for _ in range(4)],
        block_out_channels=[64, 128, 256, 256],
        layers_per_block=2,
        norm_num_groups=32,
        act_fn='silu',
        norm_type='group',
        mid_block_add_attention=True)
    if ckpt_path is not None:
        ckpt = torch.load(resolve_path(ckpt_path), map_location='cpu', weights_only=False)
        state = {k.replace('decoder.', ''): v for k, v in ckpt['state_dict'].items() if 'decoder' in k}
        decoder.load_state_dict(state, strict=True)
    return decoder


# ---------------------------------------------------------------------------------------------------------------
# upstream forward.py: block forwards without time embedding and cross-attention. As upstream, the U-Net's skip
# connections live in a module-level list, so a student network runs one forward at a time per process.
_skip = []


def _unet_forward(self, x):
    global _skip
    x = self.conv_in(x)
    _skip = [x]
    x = self.body(x)
    return x


def _cross_attn_down_block_forward(self, x):
    for i in range(2):
        x = self.resnets[i](x)
        x = self.attentions[i](x)
        _skip.append(x)
    if self.downsamplers is not None:
        x = self.downsamplers[0](x)
        _skip.append(x)
    return x


def _cross_attn_up_block_forward(self, x):
    for i in range(3):
        x = self.resnets[i](torch.cat([x, _skip.pop()], dim=1))
        x = self.attentions[i](x)
    if self.upsamplers is not None:
        x = self.upsamplers[0](x)
    return x


def _down_block_forward(self, x):
    for i in range(2):
        x = self.resnets[i](x)
        _skip.append(x)
    return x


def _mid_block_forward(self, x):
    x = self.resnets[0](x)
    x = self.attentions[0](x)
    x = self.resnets[1](x)
    return x


def _up_block_forward(self, x):
    for i in range(3):
        x = self.resnets[i](torch.cat([x, _skip.pop()], dim=1))
    x = self.upsamplers[0](x)
    return x


def _resnet_block_forward(self, x_in):
    x = self.norm1(x_in)
    x = self.nonlinearity(x)
    x = self.conv1(x)
    x = self.norm2(x)
    x = self.nonlinearity(x)
    x = self.conv2(x)
    if self.in_channels == self.out_channels:
        return x + x_in
    return x + self.conv_shortcut(x_in)


def _transformer_forward(self, x_in):
    b, c, h, w = x_in.shape
    x = self.norm(x_in)
    x = x.permute(0, 2, 3, 1).reshape(b, h * w, c).contiguous()
    x = self.proj_in(x)
    for block in self.transformer_blocks:
        x = x + block.attn1(block.norm1(x))
        x = x + block.ff(block.norm3(x))
    x = self.proj_out(x)
    x = x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    return x + x_in


# ---------------------------------------------------------------------------------------------------------------
# upstream model.py: structural compression (channel pruning to 75 % width by keeping the leading channels)
def _find_parent(model, module_name):
    components = module_name.split('.')
    parent = model
    for comp in components[:-1]:
        parent = getattr(parent, comp)
    return parent, components[-1]


def halve_channels(model):
    from diffusers.models.downsampling import Downsample2D
    from diffusers.models.upsampling import Upsample2D
    for name, module in model.named_modules():
        if hasattr(module, 'pruned'):
            continue
        if isinstance(module, nn.Conv2d):
            in_channels = int(module.in_channels * 0.75)
            out_channels = int(module.out_channels * 0.75)
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None)
            with torch.no_grad():
                new_conv.weight.copy_(module.weight[:out_channels, :in_channels])
                if module.bias is not None:
                    new_conv.bias.copy_(module.bias[:out_channels])
            parent, last_name = _find_parent(model, name)
            setattr(parent, last_name, new_conv)
            new_conv.pruned = True
        elif isinstance(module, nn.Linear):
            in_features = int(module.in_features * 0.75)
            out_features = int(module.out_features * 0.75)
            new_linear = nn.Linear(in_features=in_features, out_features=out_features, bias=module.bias is not None)
            with torch.no_grad():
                new_linear.weight.copy_(module.weight[:out_features, :in_features])
                if module.bias is not None:
                    new_linear.bias.copy_(module.bias[:out_features])
            parent, last_name = _find_parent(model, name)
            setattr(parent, last_name, new_linear)
            new_linear.pruned = True
        elif isinstance(module, nn.GroupNorm):
            num_channels = int(module.num_channels * 0.75)
            for num_groups in [32, 24, 16, 12, 8, 6, 4, 2, 1]:
                if num_channels % num_groups == 0:
                    break
            new_gn = nn.GroupNorm(
                num_groups=num_groups, num_channels=num_channels, eps=module.eps, affine=module.affine)
            with torch.no_grad():
                new_gn.weight.copy_(module.weight[:num_channels])
                new_gn.bias.copy_(module.bias[:num_channels])
            parent, last_name = _find_parent(model, name)
            setattr(parent, last_name, new_gn)
            new_gn.pruned = True
        elif isinstance(module, nn.LayerNorm):
            normalized_shape = int(module.normalized_shape[0] * 0.75)
            new_ln = nn.LayerNorm(normalized_shape, eps=module.eps, elementwise_affine=module.elementwise_affine)
            with torch.no_grad():
                new_ln.weight.copy_(module.weight[:normalized_shape])
                new_ln.bias.copy_(module.bias[:normalized_shape])
            parent, last_name = _find_parent(model, name)
            setattr(parent, last_name, new_ln)
            new_ln.pruned = True
        elif isinstance(module, Downsample2D) or isinstance(module, Upsample2D):
            module.channels = int(module.channels * 0.75)


@ARCH_REGISTRY.register()
class AdcSR(nn.Module):
    """AdcSR student network (x4 only).

    ``body`` is upstream's ``Net`` (trainable): PixelUnshuffle(2) -> SD 2.1 U-Net without time embedding and
    cross-attention, pruned to 75 % width -> mid-block of the half-width VAE decoder. ``tail`` is the frozen rest of
    that decoder (up-blocks + output conv), as in upstream ``test.py``. The latent size is LR/2 = HR/8, hence x4.
    Parameter names match upstream checkpoints (``net_params_200.pkl``) after stripping ``module.``.

    Args:
        sd_model (str): Stable Diffusion 2.1-base in diffusers layout (local folder or Hugging Face id).
        half_decoder (str): ``halfDecoder.ckpt`` (AdcSR stage 1).
        sd_init (bool): Initialise the U-Net from the SD weights (training from scratch of stage 2, as upstream).
            False only reads the U-Net config (testing: all weights come from a checkpoint). Default: True.
        color_fix (bool): Upstream test-time colour correction (per-channel mean/std of the SR image matched to the
            LR input). Default False: here LR (S2 reflectance) and HR (Maxar display RGB) differ radiometrically.

    Input: RGB in [0, 1] (clamped); output: RGB in [0, 1].
    """

    def __init__(self, sd_model, half_decoder, sd_init=True, color_fix=False):
        super().__init__()
        from diffusers.models.attention import BasicTransformerBlock
        from diffusers.models.resnet import ResnetBlock2D
        from diffusers.models.transformers.transformer_2d import Transformer2DModel
        from diffusers.models.unets.unet_2d_blocks import (CrossAttnDownBlock2D, CrossAttnUpBlock2D, DownBlock2D,
                                                           UNetMidBlock2DCrossAttn, UpBlock2D)

        self.color_fix = color_fix
        unet = load_sd_unet(sd_model, pretrained=sd_init)
        decoder = build_half_decoder(half_decoder)
        net_decoder = copy.deepcopy(decoder)

        # upstream Net.__init__
        del unet.time_embedding
        new_conv_in = nn.Conv2d(16, 320, 3, padding=1)
        new_conv_in.weight.data = unet.conv_in.weight.data.repeat(1, 4, 1, 1)
        new_conv_in.bias.data = unet.conv_in.bias.data
        unet.conv_in = new_conv_in
        new_conv_out = nn.Conv2d(320, 342, 3, padding=1)
        new_conv_out.weight.data = unet.conv_out.weight.data.repeat(86, 1, 1, 1)[:342]
        new_conv_out.bias.data = unet.conv_out.bias.data.repeat(86, )[:342]
        unet.conv_out = new_conv_out

        def remove_time_emb_proj(module):
            if isinstance(module, ResnetBlock2D):
                del module.time_emb_proj

        unet.apply(remove_time_emb_proj)

        def remove_cross_attn(module):
            if isinstance(module, BasicTransformerBlock):
                del module.attn2, module.norm2

        unet.apply(remove_cross_attn)

        def set_inplace_to_true(module):
            if isinstance(module, nn.Dropout) or isinstance(module, nn.SiLU):
                module.inplace = True

        unet.apply(set_inplace_to_true)

        def replace_forward_methods(module):
            if isinstance(module, CrossAttnDownBlock2D):
                module.forward = types.MethodType(_cross_attn_down_block_forward, module)
            elif isinstance(module, DownBlock2D):
                module.forward = types.MethodType(_down_block_forward, module)
            elif isinstance(module, UNetMidBlock2DCrossAttn):
                module.forward = types.MethodType(_mid_block_forward, module)
            elif isinstance(module, UpBlock2D):
                module.forward = types.MethodType(_up_block_forward, module)
            elif isinstance(module, CrossAttnUpBlock2D):
                module.forward = types.MethodType(_cross_attn_up_block_forward, module)
            elif isinstance(module, ResnetBlock2D):
                module.forward = types.MethodType(_resnet_block_forward, module)
            elif isinstance(module, Transformer2DModel):
                module.forward = types.MethodType(_transformer_forward, module)

        unet.apply(replace_forward_methods)
        unet.forward = types.MethodType(_unet_forward, unet)
        halve_channels(unet)
        unet.body = nn.Sequential(
            *unet.down_blocks,
            unet.mid_block,
            *unet.up_blocks,
            unet.conv_norm_out,
            unet.conv_act,
            unet.conv_out,
        )
        del net_decoder.conv_in, net_decoder.up_blocks, net_decoder.conv_norm_out, net_decoder.conv_act
        del net_decoder.conv_out
        self.body = nn.Sequential(
            nn.PixelUnshuffle(2),
            unet,
            net_decoder.mid_block,
        )

        # frozen decoder tail (upstream test.py)
        self.tail = nn.Sequential(*decoder.up_blocks, decoder.conv_norm_out, decoder.conv_act, decoder.conv_out)
        self.tail.requires_grad_(False)

    def forward(self, x, features=False):
        """``features=True`` returns the student output in the teacher's feature space (after the decoder mid-block),
        which is what AdcSR is trained on; otherwise the SR image."""
        lr = x.clamp(0, 1) * 2 - 1
        feat = self.body(lr)
        if features:
            return feat
        sr = self.tail(feat)
        if self.color_fix:
            sr = (sr - sr.mean(dim=[2, 3], keepdim=True)) / sr.std(dim=[2, 3], keepdim=True) \
                * lr.std(dim=[2, 3], keepdim=True) + lr.mean(dim=[2, 3], keepdim=True)
        return sr / 2 + 0.5


@ARCH_REGISTRY.register()
class AdcSRDiscriminator(nn.Module):
    """AdcSR discriminator (upstream ``train.py``): a copy of the SD 2.1 U-Net with a new input conv for the
    256-channel decoder features and LoRA adapters on its conv/linear layers. Only the LoRA weights and the input conv
    are trained; the rest stays at the SD weights. Called with the fixed timestep 999 and the prompt embedding.

    Args:
        sd_model (str): Stable Diffusion 2.1-base (diffusers layout).
        num_in_ch (int): Feature channels of the decoder mid-block. Default: 256.
        lora_rank (int): LoRA rank. Default: 4.
    """

    def __init__(self, sd_model, num_in_ch=256, lora_rank=4):
        super().__init__()
        from peft import LoraConfig

        unet = load_sd_unet(sd_model)
        new_conv_in = nn.Conv2d(num_in_ch, 320, 3, padding=1)
        new_conv_in.weight.data = unet.conv_in.weight.data.repeat(1, num_in_ch // 4, 1, 1) / (num_in_ch // 4)
        new_conv_in.bias.data = unet.conv_in.bias.data
        unet.conv_in = new_conv_in

        # upstream utils.add_lora_to_unet
        encoder, decoder, others = [], [], []
        patterns = [
            'to_k', 'to_q', 'to_v', 'to_out.0', 'conv', 'conv1', 'conv2', 'conv_shortcut', 'conv_out', 'proj_out',
            'proj_in', 'ff.net.2', 'ff.net.0.proj'
        ]
        for n, _ in unet.named_parameters():
            if 'bias' in n or 'norm' in n:
                continue
            for pattern in patterns:
                if pattern in n and ('down_blocks' in n or 'conv_in' in n):
                    encoder.append(n.replace('.weight', ''))
                    break
                elif pattern in n and ('up_blocks' in n or 'conv_out' in n):
                    decoder.append(n.replace('.weight', ''))
                    break
                elif pattern in n:
                    others.append(n.replace('.weight', ''))
                    break
        for name, modules in [('default_encoder', encoder), ('default_decoder', decoder), ('default_others', others)]:
            unet.add_adapter(
                LoraConfig(r=lora_rank, init_lora_weights='gaussian', target_modules=modules), adapter_name=name)
        unet.set_adapters(['default_encoder', 'default_decoder', 'default_others'])

        unet.requires_grad_(False)
        for n, p in unet.named_parameters():
            if 'lora' in n or 'conv_in' in n:
                p.requires_grad = True
        self.unet = unet

    def trainable_state_dict(self):
        """Only the trained weights (LoRA + input conv); the frozen SD weights are rebuilt from ``sd_model``."""
        return {n: p.detach() for n, p in self.named_parameters() if p.requires_grad}

    def forward(self, x, timesteps, encoder_hidden_states):
        return self.unet(x, timesteps, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
