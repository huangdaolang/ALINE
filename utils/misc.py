import torch
from torch import optim
from torch.optim import lr_scheduler
import os
import numpy as np
import random
from omegaconf import OmegaConf
import hydra
from hydra import initialize_config_dir, compose
from model.base import Aline

def set_seed(seed):
    """
    Sets the seed for generating random numbers across several libraries to ensure reproducibility.
    This function sets the random seed for PyTorch, PyTorch's CUDA backend, NumPy, and Python's
    random module. It also configures PyTorch's backend to enforce deterministic behavior.

    Args:
        seed (int): The seed value for random number generation.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_state_dict(model, dir, name="aline.pth"):
    """
    Saves the state dictionary of a PyTorch model to a specified directory with a given filename.

    Args:
        model (torch.nn.Module): The PyTorch model whose state dictionary is to be saved.
        dir (str): The directory where the model state dictionary will be saved.
        name (str, optional): The filename for the saved state dictionary. Defaults to "aline.pth".
    """
    file_path = os.path.join(dir, 'model')
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_path = os.path.join(file_path, name)
    torch.save(model.state_dict(), file_path)
    return file_path


def load_state_dict(model, dir, name="aline.pth"):
    """
    Loads the state dictionary of a PyTorch model from a specified directory and filename.

    Args:
        model (torch.nn.Module): The PyTorch model into which the state dictionary is to be loaded.
        dir (str): The directory from which the model state dictionary will be loaded.
        name (str, optional): The filename of the saved state dictionary. Defaults to "aline.pth".
    """
    file_path = os.path.join(dir, 'model', name)
    model.load_state_dict(torch.load(file_path, map_location=torch.get_default_device(), weights_only=True))
    return model


def save_checkpoint(cfg, model, optimizer, scheduler, epoch, with_epoch=False):
    """ Save checkpoint

    Args:
        cfg (AttrDict): config
        model (nn.Module): model
        optimizer (torch.optim.Optimizer): optimizer
        scheduler (torch.optim.Scheduler): scheduler
        epoch (int): current epoch
        with_epoch (bool, optional): whether to add epoch as suffix. e.g. ckptname_1000.tar
    """
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        # random generator
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        'numpy_rng_state': np.random.get_state(),
        'random_rng_state': random.getstate(),
    }

    if with_epoch:
        checkpoint_name = f"{cfg.checkpoint_name.split('.')[0]}_{epoch}.tar"
    else:
        checkpoint_name = cfg.checkpoint_name

    checkpoint_path = os.path.join(cfg.output_dir, checkpoint_name)
    torch.save(ckpt, str(checkpoint_path))


def load_checkpoint(cfg, model, optimizer, scheduler, ckpt_path=None, check_layerwise=True):
    """ Load checkpoint

    Args:
        cfg (AttrDict): config
        model (nn.Module): model
        optimizer (torch.optim.Optimizer): optimizer
        scheduler (torch.optim.Scheduler): scheduler
        ckpt_path (str): specified checkpoint path. Defaults to None.
        check_layerwise (bool): set layerwise optimizer.

    Returns:
        epoch (int): current epoch
        optimizer (torch.optim.Optimizer): optimizer
        scheduler (torch.optim.Scheduler): scheduler
    """
    if ckpt_path is None:
        ckpt_path = os.path.join(cfg.output_dir, cfg.checkpoint_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=torch.get_default_device(), weights_only=False)
    model.load_state_dict(ckpt["model"])
    epoch = ckpt["epoch"]

    if check_layerwise:
        optimizer, scheduler = set_layerwise_lr(cfg, model, epoch - 1) # optimizer and sheduler state in the last epoch

    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    # Restore RNG states
    if not isinstance(ckpt['rng_state'], torch.ByteTensor):
        ckpt['rng_state'] = torch.ByteTensor(ckpt['rng_state'].cpu())
    torch.set_rng_state(ckpt['rng_state'])

    if torch.cuda.is_available() and ckpt['cuda_rng_state'] is not None:
        if not isinstance(ckpt['cuda_rng_state'], torch.ByteTensor):
            ckpt['cuda_rng_state'] = torch.ByteTensor(ckpt['cuda_rng_state'].cpu())
        torch.cuda.set_rng_state(ckpt['cuda_rng_state'])

    np.random.set_state(ckpt['numpy_rng_state'])
    random.setstate(ckpt['random_rng_state'])

    return epoch, optimizer, scheduler

def set_layerwise_lr(cfg, model, epoch=0):
    """ Set layerwise learning rate

    Args:
        cfg (AttrDict): config
        model (nn.Module): model
        epoch (int, optional): current epoch. Defaults to 0.
    """
    if epoch < cfg.burning_epoch:
        optimizer = getattr(optim, cfg.optimizer)(
            model.parameters(),
            lr=cfg.lr,
        )
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.max_epoch
        )    
    else:
        shared_params, predictor_params = [], []
        for name, param in model.named_parameters():
            if 'predictor' not in name:
                shared_params.append(param)
            else:
                predictor_params.append(param)

        optimizer = getattr(optim, cfg.optimizer)(
            [
                {"params": shared_params, "lr": cfg.lr / 5},
                {"params": predictor_params},
            ],
            lr=cfg.lr,
        )
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.max_epoch - cfg.burning_epoch
        )
    return optimizer, scheduler


def load_config_and_model(path, config_name="config.yaml", file_name="aline.pth", 
                          load_type="ckpt", device=None):
    """
    Loads configuration and model from a specified path.
    
    Args:
        path (str): Path to the model directory (relative to project root)
        config_name (str): Name of the config file
        file_name (str): Name of the model file
        load_type (str): "ckpt" or "pth"
        device (str): Device to load model on (if None, uses default)
    
    Returns:
        tuple: (resolved_config, model)
    """

    # Normalize to absolute path based on project root (parent of utils)
    if os.path.isabs(path):
        full_dir = path
    else:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        full_dir = os.path.abspath(os.path.join(project_root, path))

    config_dir = os.path.join(full_dir, ".hydra")

    if not os.path.isdir(config_dir):
        raise FileNotFoundError(f"Config path not found: {config_dir}")

    # Use absolute config directory to avoid Hydra treating it as relative to module path
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)
        resolved_cfg = OmegaConf.create(OmegaConf.to_yaml(cfg, resolve=True))

        if device is not None:
            torch.set_default_device(device)

        # Instantiate model components
        embedder = hydra.utils.instantiate(cfg.embedder)
        encoder = hydra.utils.instantiate(cfg.encoder)
        head = hydra.utils.instantiate(cfg.head)
        model = Aline(embedder, encoder, head)

        # Load model weights using resolved absolute path
        file_path = os.path.join(full_dir, file_name)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        try:
            if load_type.lower() == "ckpt":
                ckpt = torch.load(file_path, map_location=torch.get_default_device(), weights_only=False)
                
                if isinstance(ckpt, dict) and "model" in ckpt:
                    model.load_state_dict(ckpt["model"])
                else:
                    model.load_state_dict(ckpt)
                    
            elif load_type.lower() == "pth":
                state_dict = torch.load(file_path, map_location=torch.get_default_device(), weights_only=True)
                model.load_state_dict(state_dict)
                
            else:
                raise ValueError(f"Invalid load_type: {load_type}. Must be 'ckpt' or 'pth'")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {file_path}: {str(e)}")
    
    return resolved_cfg, model


def calculate_gmm_variance(mixture_means, mixture_stds, mixture_weights):
    """
    Calculate the uncertainty of GMM predictions for each query point.

    Args:
        mixture_means: [batch_size, n_query, n_components] tensor of means
        mixture_stds: [batch_size, n_query, n_components] tensor of standard deviations
        mixture_weights: [batch_size, n_query, n_components] or [batch_size, n_components] weights

    Returns:
        variance: [batch_size, n_query] tensor of variance
    """
    batch_size, n_query, n_components = mixture_means.shape

    # Adjust weights shape if necessary
    if len(mixture_weights.shape) == 2:  # [batch_size, n_components]
        # Expand to [batch_size, n_query, n_components]
        weights = mixture_weights.unsqueeze(1).expand(batch_size, n_query, n_components)
    else:
        weights = mixture_weights

    # Calculate weighted mean for each query point
    # Sum along component dimension (dim=2)
    weighted_means = torch.sum(weights * mixture_means, dim=2)  # [batch_size, n_query]

    # Calculate variance for GMM: var = Σ weight * (std^2 + (mean - weighted_mean)^2)
    # First, calculate (mean - weighted_mean)^2
    mean_diffs_squared = (mixture_means - weighted_means.unsqueeze(2)) ** 2  # [batch_size, n_query, n_components]

    # Calculate total variance
    var = torch.sum(
        weights * (mixture_stds ** 2 + mean_diffs_squared),
        dim=2
    )  # [batch_size, n_query]

    return var

