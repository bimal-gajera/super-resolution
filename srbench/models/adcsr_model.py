"""AdcSR training (stage 2 of Chen et al., CVPR 2025), ported from upstream ``train.py``
(https://github.com/Guaishou74851/AdcSR, Apache-2.0, commit d0b2871).

Each iteration, under fp16 autocast as upstream:

* frozen: DAPE/RAM tags of the LR image -> SD 2.1 text embedding (conditions teacher and discriminator); the OSEDiff
  teacher (one step at t = 999 on the bicubic-upsampled LR) -> its x0 estimate -> half-decoder mid-block features;
  the SD VAE encoding of the GT -> the same features;
* student (``AdcSR`` arch, trained on those features): L1 to the teacher features + non-saturating adversarial loss;
* discriminator (``AdcSRDiscriminator``, LoRA on SD 2.1): GT features real, student features fake.

``adcsr_opt.distill_target: gt`` replaces the teacher features by the GT features (no OSEDiff needed). Needs the
``envs/adcsr`` environment. Differences from upstream: separate loss scalers for G and D (upstream shares one), and the
scaler state is not saved in the training state (it re-calibrates within a few iterations after a resume).
"""
import os
import torch
from collections import OrderedDict
from torch.nn import functional as F

from srbench.archs import build_network
from srbench.archs.adcsr_arch import build_half_decoder, resolve_path
from srbench.utils import get_root_logger
from srbench.utils.dist_util import master_only
from srbench.utils.registry import MODEL_REGISTRY
from .base_model import _atomic_save
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class AdcSRModel(SRModel):
    """AdcSR adversarial distillation (student = ``network_g``, discriminator = ``network_d``)."""

    def init_training_settings(self):
        train_opt = self.opt['train']
        adcsr_opt = train_opt['adcsr_opt']
        logger = get_root_logger()
        if train_opt.get('ema_decay', 0) > 0:
            raise ValueError('AdcSRModel does not use EMA (upstream trains without it): set train.ema_decay: 0')
        self.ema_decay = 0
        self.distill_target = adcsr_opt.get('distill_target', 'teacher')
        assert self.distill_target in ('teacher', 'gt'), f'unknown distill_target {self.distill_target}'
        self.distill_weight = adcsr_opt.get('distill_weight', 1.0)
        self.adv_weight = adcsr_opt.get('adv_weight', 1.0)
        self.amp = adcsr_opt.get('amp', True)

        self.net_g.train()
        self.net_d = self.model_to_device(build_network(self.opt['network_d']))
        self.net_d.train()
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            self._load_trainable(self.net_d, load_path)
        n_d = sum(p.numel() for p in self.net_d.parameters() if p.requires_grad)
        logger.info(f'Discriminator: {n_d / 1e6:.2f}M trainable parameters (LoRA + input conv).')

        self._build_frozen(adcsr_opt)

        self.setup_optimizers()
        self.setup_schedulers()  # generator only, as upstream (the discriminator learning rate stays constant)
        optim_opt = dict(train_opt['optim_d'])
        optim_type = optim_opt.pop('type')
        params_d = [p for p in self.net_d.parameters() if p.requires_grad]
        self.optimizer_d = self.get_optimizer(optim_type, params_d, **optim_opt)
        self.optimizers.append(self.optimizer_d)
        self.scaler_g = torch.amp.GradScaler('cuda', enabled=self.amp)
        self.scaler_d = torch.amp.GradScaler('cuda', enabled=self.amp)

    def _build_frozen(self, adcsr_opt):
        """SD VAE, text encoder, DAPE/RAM prompt extractor, half-decoder and (optionally) the OSEDiff teacher."""
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from torchvision import transforms
        from transformers import CLIPTextModel, CLIPTokenizer

        from srbench.third_party.ram.models.ram_lora import ram

        logger = get_root_logger()
        sd_model = resolve_path(adcsr_opt['sd_model'])
        self.vae = AutoencoderKL.from_pretrained(sd_model, subfolder='vae').to(self.device).requires_grad_(False)
        self.tokenizer = CLIPTokenizer.from_pretrained(sd_model, subfolder='tokenizer')
        self.text_encoder = CLIPTextModel.from_pretrained(
            sd_model, subfolder='text_encoder').to(self.device).requires_grad_(False)
        self.alpha = DDPMScheduler.from_pretrained(sd_model, subfolder='scheduler').alphas_cumprod[999].item()
        self.decoder = build_half_decoder(self.opt['network_g']['half_decoder']).to(self.device).requires_grad_(False)
        self.dape = ram(
            pretrained=resolve_path(adcsr_opt['ram']),
            pretrained_condition=resolve_path(adcsr_opt['dape']),
            image_size=384,
            vit='swin_l').eval().to(self.device).requires_grad_(False)
        self.ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if self.distill_target == 'teacher':
            osediff = torch.load(resolve_path(adcsr_opt['osediff']), map_location='cpu', weights_only=False)
            self.vae_teacher = AutoencoderKL.from_pretrained(sd_model, subfolder='vae')
            self.vae_teacher.load_state_dict(osediff['vae'])
            self.unet_teacher = UNet2DConditionModel.from_pretrained(sd_model, subfolder='unet')
            self.unet_teacher.load_state_dict(osediff['unet'])
            self.vae_teacher.to(self.device).requires_grad_(False)
            self.unet_teacher.to(self.device).requires_grad_(False)
            del osediff
        logger.info(f'AdcSR frozen components loaded (distillation target: {self.distill_target}).')

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_opt = dict(train_opt['optim_g'])
        optim_type = optim_opt.pop('type')
        params_g = [p for p in self.net_g.parameters() if p.requires_grad]
        self.optimizer_g = self.get_optimizer(optim_type, params_g, **optim_opt)
        self.optimizers.append(self.optimizer_g)

    def _decoder_features(self, vae, latents):
        """Latents (without the SD scaling factor) -> features after the half-decoder mid-block."""
        return self.decoder.mid_block(self.decoder.conv_in(vae.post_quant_conv(latents)))

    @torch.no_grad()
    def _targets(self, timesteps):
        """Prompt embedding, GT features and distillation target for the current batch."""
        lq = self.lq.clamp(0, 1)
        tags = self.dape.generate_tag(self.ram_transforms(lq))[0]
        tokens = self.tokenizer(
            tags,
            max_length=self.tokenizer.model_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt').input_ids.to(self.device)
        text = self.text_encoder(tokens, return_dict=False)[0]
        z0_gt = self._decoder_features(self.vae, self.vae.encode(self.gt * 2 - 1).latent_dist.mean)
        if self.distill_target == 'gt':
            return text, z0_gt, z0_gt
        scaling = self.vae_teacher.config.scaling_factor
        lr_up = F.interpolate(lq * 2 - 1, scale_factor=self.opt['scale'], mode='bicubic')
        latents = self.vae_teacher.encode(lr_up).latent_dist.mean * scaling
        eps = self.unet_teacher(latents, timesteps, encoder_hidden_states=text, return_dict=False)[0]
        z0 = (latents - (1 - self.alpha)**0.5 * eps) / self.alpha**0.5
        return text, z0_gt, self._decoder_features(self.vae_teacher, z0 / scaling)

    def optimize_parameters(self, current_iter):
        timesteps = torch.full((self.lq.size(0), ), 999, device=self.device, dtype=torch.long)
        loss_dict = OrderedDict()
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.amp):
            text, z0_gt, target = self._targets(timesteps)

            # generator (the discriminator's gradients from this step are discarded by its zero_grad below, as
            # upstream; its trainable parameters keep requires_grad=True so that DDP sees a fixed parameter set)
            self.optimizer_g.zero_grad(set_to_none=True)
            z0_student = self.net_g(self.lq, features=True)
            l_distil = self.distill_weight * (z0_student - target).abs().mean()
            l_adv = self.adv_weight * F.softplus(-self.net_d(z0_student, timesteps, text)).mean()
            l_g_total = l_distil + l_adv
        self.scaler_g.scale(l_g_total).backward()
        self.scaler_g.step(self.optimizer_g)
        self.scaler_g.update()
        loss_dict['l_g_distil'] = l_distil
        loss_dict['l_g_adv'] = l_adv

        # discriminator
        self.optimizer_d.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.amp):
            pred_real = self.net_d(z0_gt, timesteps, text)
            pred_fake = self.net_d(z0_student.detach(), timesteps, text)
            l_d = F.softplus(pred_fake).mean() + F.softplus(-pred_real).mean()
        self.scaler_d.scale(l_d).backward()
        self.scaler_d.step(self.optimizer_d)
        self.scaler_d.update()
        loss_dict['l_d'] = l_d
        loss_dict['out_d_real'] = pred_real.detach().float().mean()
        loss_dict['out_d_fake'] = pred_fake.detach().float().mean()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        if current_iter in (1, self.opt['logger']['print_freq']):
            get_root_logger().info(f'Peak GPU memory after iteration {current_iter}: '
                                   f'{torch.cuda.max_memory_allocated() / 2**30:.1f} GB')

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self._save_trainable(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)

    @master_only
    def _save_trainable(self, net, net_label, current_iter):
        """Save only the trained discriminator weights (LoRA + input conv, a few MB instead of 3.5 GB)."""
        label = 'latest' if current_iter == -1 else current_iter
        state = {k: v.cpu() for k, v in self.get_bare_model(net).trainable_state_dict().items()}
        _atomic_save({'params': state}, os.path.join(self.opt['path']['models'], f'{net_label}_{label}.pth'))
        self._prune_checkpoints(self.opt['path']['models'], f'{net_label}_*.pth')

    def _load_trainable(self, net, load_path):
        net = self.get_bare_model(net)
        state = torch.load(load_path, map_location='cpu', weights_only=True)['params']
        expected = set(net.trainable_state_dict())
        missing, unexpected = expected - set(state), set(state) - expected
        if missing or unexpected:
            raise KeyError(f'{load_path}: {len(missing)} missing / {len(unexpected)} unexpected discriminator keys')
        net.load_state_dict(state, strict=False)
        get_root_logger().info(f'Loading {net.__class__.__name__} trainable weights from {load_path}.')
