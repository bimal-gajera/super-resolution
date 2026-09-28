from copy import deepcopy

from srbench.utils.registry import METRIC_REGISTRY
from .lpips_metric import calculate_lpips
from .psnr_ssim import calculate_psnr, calculate_ssim

__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_lpips']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    opt.pop('better', None)  # only used for best-metric tracking
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
