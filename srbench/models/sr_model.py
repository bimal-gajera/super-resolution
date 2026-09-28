import csv
import torch
import torch.nn.functional as F
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from srbench.archs import build_network
from srbench.losses import build_loss
from srbench.metrics import calculate_metric
from srbench.utils import get_root_logger, imwrite, tensor2img
from srbench.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device, non_blocking=True)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device, non_blocking=True)
        self.lq_rgb = data.get('lq_rgb')  # (cpu) S2 RGB composite, only used for visualisation

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def test_selfensemble(self):
        # 8 augmentations (flips + transpose), modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            if op == 'v':
                return v.flip(-1)
            if op == 'h':
                return v.flip(-2)
            return v.transpose(-2, -1)  # 't'

        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        net = self.net_g_ema if hasattr(self, 'net_g_ema') else self.net_g
        net.eval()
        with torch.no_grad():
            out_list = [net(aug) for aug in lq_list]
        if not hasattr(self, 'net_g_ema'):
            self.net_g.train()

        # undo the transforms
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        self.output = torch.stack(out_list, dim=0).mean(dim=0)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """Validation / test loop (srbench: batched; per-image metrics; LR|SR|GT panels; best checkpoint).

        Extra ``val`` options on top of BasicSR:
            max_save_img (int | None): Save images for the first N samples only (default: all when save_img).
            save_panel (bool): Save LR|SR|GT comparison panels. Default: True.
            tb_images (int): Log the first N panels to tensorboard. Default: 4.
            save_best (str | None): Metric name; keep ``net_g_best.pth`` for the best value on the first
                validation set. Default: None.
            self_ensemble (bool): x8 geometric self-ensemble at test time. Default: False.
        """
        dataset_name = dataloader.dataset.opt['name']
        val_opt = self.opt['val']
        with_metrics = val_opt.get('metrics') is not None
        use_pbar = val_opt.get('pbar', False)
        max_save = val_opt.get('max_save_img')
        save_panel = val_opt.get('save_panel', True)
        tb_images = val_opt.get('tb_images', 4) if tb_logger is not None else 0

        if with_metrics:
            self._initialize_best_metric_results(dataset_name)
            metric_sums = {metric: 0. for metric in val_opt['metrics']}
        per_image, n_img = [], 0
        pbar = tqdm(total=len(dataloader.dataset), unit='image') if use_pbar else None

        for val_data in dataloader:
            self.feed_data(val_data)
            self.test_selfensemble() if val_opt.get('self_ensemble', False) else self.test()
            output = self.output.detach().float().cpu()
            gts = val_data.get('gt')
            for b in range(output.size(0)):
                key = val_data['key'][b]
                sr_img = tensor2img(output[b])
                metric_data = {'img': sr_img}
                if gts is not None:
                    metric_data['img2'] = tensor2img(gts[b])
                row = {'key': key}
                if with_metrics:
                    for name, opt_ in val_opt['metrics'].items():
                        value = calculate_metric(metric_data, opt_)
                        metric_sums[name] += value
                        row[name] = value
                per_image.append(row)

                want_img = save_img and (max_save is None or n_img < max_save)
                if want_img or n_img < tb_images:
                    panel = self._make_panel(b, output[b], gts[b] if gts is not None else None)
                    if want_img:
                        self._save_images(sr_img, panel if save_panel else None, key, dataset_name, current_iter)
                    if n_img < tb_images:
                        tb_logger.add_image(f'val_{dataset_name}/{key}', panel, current_iter, dataformats='HWC')
                n_img += 1
                if pbar is not None:
                    pbar.update(1)
            # free GPU memory
            del self.lq, self.output
            if hasattr(self, 'gt'):
                del self.gt
        if pbar is not None:
            pbar.close()

        if with_metrics:
            self.metric_results = {metric: total / max(n_img, 1) for metric, total in metric_sums.items()}
            for metric, value in self.metric_results.items():
                self._update_best_metric_result(dataset_name, metric, value, current_iter)
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            self._maybe_save_best(dataset_name, current_iter)

        if not self.opt['is_train']:  # per-image metrics for later analysis
            csv_path = osp.join(self.opt['path']['results_root'], f'metrics_{dataset_name}.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(per_image[0].keys()) if per_image else ['key'])
                writer.writeheader()
                writer.writerows(per_image)
            get_root_logger().info(f'Per-image metrics written to {csv_path}')

    def _make_panel(self, b, sr, gt):
        """RGB float [0, 1] HWC panel: S2 RGB (nearest-upsampled) | SR | GT."""
        h, w = sr.shape[-2:]
        tiles = []
        if self.lq_rgb is not None:
            tiles.append(F.interpolate(self.lq_rgb[b:b + 1].float(), size=(h, w), mode='nearest')[0])
        tiles.append(sr.clamp(0, 1))
        if gt is not None:
            tiles.append(gt.clamp(0, 1))
        sep = torch.ones(3, h, 4)
        panel = torch.cat([t for tile in tiles for t in (tile, sep)][:-1], dim=2)
        return panel.permute(1, 2, 0).numpy()

    def _save_images(self, sr_img, panel, key, dataset_name, current_iter):
        vis_root = self.opt['path']['visualization']
        if self.opt['is_train']:
            if panel is not None:  # LR | SR | GT is more informative than SR alone while training
                imwrite(tensor2img(torch.from_numpy(panel).permute(2, 0, 1)), osp.join(vis_root, key,
                                                                                       f'{key}_{current_iter}.png'))
            else:
                imwrite(sr_img, osp.join(vis_root, key, f'{key}_{current_iter}.png'))
            return
        suffix = self.opt['val'].get('suffix') or self.opt['name']
        imwrite(sr_img, osp.join(vis_root, dataset_name, f'{key}_{suffix}.png'))
        if panel is not None:
            imwrite(
                tensor2img(torch.from_numpy(panel).permute(2, 0, 1)),
                osp.join(vis_root, f'{dataset_name}_panels', f'{key}.png'))

    def _maybe_save_best(self, dataset_name, current_iter):
        metric = self.opt['val'].get('save_best')
        if not metric or not self.opt['is_train']:
            return
        if not hasattr(self, '_best_dataset'):
            self._best_dataset = dataset_name  # only the first validation set drives the best checkpoint
        if dataset_name != self._best_dataset:
            return
        best = self.best_metric_results[dataset_name][metric]
        if best['iter'] == current_iter:
            net = self.net_g_ema if hasattr(self, 'net_g_ema') else self.net_g
            self.save_network(net, 'net_g', 'best')
            get_root_logger().info(f'Saved net_g_best.pth ({metric} {best["val"]:.4f} @ iter {current_iter})')

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
