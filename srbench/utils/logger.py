import datetime
import logging
import time
import uuid
from os import path as osp

from .dist_util import get_dist_info, master_only

initialized_logger = {}


class AvgTimer():

    def __init__(self, window=200):
        self.window = window  # average window
        self.current_time = 0
        self.total_time = 0
        self.count = 0
        self.avg_time = 0
        self.start()

    def start(self):
        self.start_time = self.tic = time.time()

    def record(self):
        self.count += 1
        self.toc = time.time()
        self.current_time = self.toc - self.tic
        self.total_time += self.current_time
        # calculate average time
        self.avg_time = self.total_time / self.count

        # reset
        if self.count > self.window:
            self.count = 0
            self.total_time = 0

        self.tic = time.time()

    def get_current_time(self):
        return self.current_time

    def get_avg_time(self):
        return self.avg_time


class MessageLogger():
    """Message logger for printing.

    Args:
        opt (dict): Config. It contains the following keys:
            name (str): Exp name.
            logger (dict): Contains 'print_freq' (str) for logger interval.
            train (dict): Contains 'total_iter' (int) for total iters.
            use_tb_logger (bool): Use tensorboard logger.
        start_iter (int): Start iter. Default: 1.
        tb_logger (obj:`tb_logger`): Tensorboard logger. Default： None.
    """

    def __init__(self, opt, start_iter=1, tb_logger=None):
        self.exp_name = opt['name']
        self.interval = opt['logger']['print_freq']
        self.start_iter = start_iter
        self.max_iters = opt['train']['total_iter']
        self.use_tb_logger = opt['logger']['use_tb_logger']
        self.tb_logger = tb_logger
        self.start_time = time.time()
        self.logger = get_root_logger()

    def reset_start_time(self):
        self.start_time = time.time()

    @master_only
    def __call__(self, log_vars):
        """Format logging message.

        Args:
            log_vars (dict): It contains the following keys:
                epoch (int): Epoch number.
                iter (int): Current iter.
                lrs (list): List for learning rates.

                time (float): Iter time.
                data_time (float): Data time for each iter.
        """
        # epoch, iter, learning rates
        epoch = log_vars.pop('epoch')
        current_iter = log_vars.pop('iter')
        lrs = log_vars.pop('lrs')

        message = (f'[{self.exp_name[:5]}..][epoch:{epoch:3d}, iter:{current_iter:8,d}, lr:(')
        for v in lrs:
            message += f'{v:.3e},'
        message += ')] '

        # time and estimated time
        if 'time' in log_vars.keys():
            iter_time = log_vars.pop('time')
            data_time = log_vars.pop('data_time')

            total_time = time.time() - self.start_time
            time_sec_avg = total_time / (current_iter - self.start_iter + 1)
            eta_sec = time_sec_avg * max(self.max_iters - current_iter, 0)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            message += f'[eta: {eta_str}, '
            message += f'time (data): {iter_time:.3f} ({data_time:.3f})] '

        # other items, especially losses
        for k, v in log_vars.items():
            message += f'{k}: {v:.4e} '
            # tensorboard logger
            if self.use_tb_logger and 'debug' not in self.exp_name:
                if k.startswith('l_'):
                    self.tb_logger.add_scalar(f'losses/{k}', v, current_iter)
                else:
                    self.tb_logger.add_scalar(k, v, current_iter)
        self.logger.info(message)


@master_only
def init_tb_logger(log_dir):
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = SummaryWriter(log_dir=log_dir)
    return tb_logger


@master_only
def init_wandb_logger(opt):
    """Weights & Biases logging (mirrors the tensorboard logs via ``sync_tensorboard``).

    srbench: the run id is stored in ``<experiment>/wandb_id.txt`` when the run is created, so jobs continued with
    ``--auto_resume`` (e.g. chained Slurm jobs) append to the same W&B run instead of starting a new one.
    ``logger.wandb.resume_id`` still overrides it. Optional keys: ``entity``, ``group``, ``tags``.
    Offline use: ``export WANDB_MODE=offline`` and later ``wandb sync <experiment>/wandb/offline-run-*``.
    """
    import wandb
    logger = get_root_logger()

    wandb_opt = opt['logger']['wandb']
    exp_root = opt['path']['experiments_root']
    id_file = osp.join(exp_root, 'wandb_id.txt')
    wandb_id = wandb_opt.get('resume_id')
    if not wandb_id and osp.exists(id_file):
        with open(id_file) as f:
            wandb_id = f.read().strip()
    if wandb_id:
        logger.info(f'Resume wandb run id={wandb_id}.')
    else:
        wandb_id = uuid.uuid4().hex[:8]
    with open(id_file, 'w') as f:
        f.write(wandb_id + '\n')

    wandb.init(
        id=wandb_id,
        resume='allow',
        name=opt['name'],
        project=wandb_opt['project'],
        entity=wandb_opt.get('entity'),
        group=wandb_opt.get('group'),
        tags=wandb_opt.get('tags'),
        config=opt,
        dir=exp_root,
        sync_tensorboard=True)
    logger.info(f'Use wandb logger with id={wandb_id}; project={wandb_opt["project"]}.')


def finish_wandb_logger(opt):
    """Close the W&B run (flushes the last tensorboard events) if one was started."""
    if opt['logger'].get('wandb', {}).get('project') and 'debug' not in opt['name'] and opt.get('rank', 0) == 0:
        import wandb
        if wandb.run is not None:
            wandb.finish()


def get_root_logger(logger_name='srbench', log_level=logging.INFO, log_file=None):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.

    Args:
        logger_name (str): root logger name. Default: 'srbench'.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger_name in initialized_logger:
        return logger

    format_str = '%(asctime)s %(levelname)s: %(message)s'
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    elif log_file is not None:
        logger.setLevel(log_level)
        # add file handler
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    initialized_logger[logger_name] = True
    return logger


def get_env_info():
    """Get environment information (software versions and GPUs)."""
    import torch
    import torchvision

    from srbench import __version__
    msg = ('\nVersion Information: '
           f'\n\tsrbench: {__version__}'
           f'\n\tPyTorch: {torch.__version__} (CUDA {torch.version.cuda})'
           f'\n\tTorchVision: {torchvision.__version__}')
    if torch.cuda.is_available():
        msg += '\n\tGPUs: ' + ', '.join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
    return msg
