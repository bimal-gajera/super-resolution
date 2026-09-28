import numpy as np
import torch

from srbench.utils.registry import METRIC_REGISTRY

_LPIPS_MODELS = {}


def _get_lpips(net, device):
    key = (net, str(device))
    if key not in _LPIPS_MODELS:
        import lpips  # optional dependency, imported lazily
        _LPIPS_MODELS[key] = lpips.LPIPS(net=net, verbose=False).to(device).eval()
    return _LPIPS_MODELS[key]


@METRIC_REGISTRY.register()
def calculate_lpips(img, img2, crop_border=0, net='alex', input_order='HWC', **kwargs):
    """LPIPS perceptual distance (lower is better).

    Args:
        img (ndarray): uint8 image in [0, 255], BGR, HWC (as returned by ``tensor2img``).
        img2 (ndarray): Reference image, same format.
        crop_border (int): Pixels cropped from each edge before computing the metric.
        net (str): LPIPS backbone, 'alex' (default, standard for SR) or 'vgg'.

    Returns:
        float: LPIPS distance.
    """
    assert img.shape == img2.shape, f'Image shapes are different: {img.shape}, {img2.shape}.'
    assert input_order == 'HWC'
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = _get_lpips(net, device)

    def _to_tensor(x):  # BGR uint8 HWC -> RGB float NCHW in [-1, 1]
        x = np.ascontiguousarray(x[..., ::-1]).astype(np.float32) / 127.5 - 1
        return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        return model(_to_tensor(img), _to_tensor(img2)).item()
