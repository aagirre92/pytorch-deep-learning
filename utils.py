
"""
File containing various utility functions
"""
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter

def set_seeds_cpu_cuda(seed: int = 42):
    """Sets random seeds for torch operations

    Args:
      seed(int,optional): Random seed to set, 42 by default
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print(f"Seed set to {seed}")


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
      model: A target PyTorch model to save.
      target_dir: A directory for saving the model to.
      model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.

    Example usage:
      save_model(model=model_0,
                 target_dir="models",
                 model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)


def create_summary_writer(experiment_name: str,
                          model_name: str,
                          extra: str = None):
    """
    Creates a torch.utils.tensorboard.writer.SummaryWriter() instance tracking to a specific directory.
    """

    from datetime import datetime
    import os

    # Get timestamp of current date
    timestamp = datetime.now().strftime("%Y-%m-%d")

    if extra:
        # Create log directory path
        log_dir = os.path.join(
            "runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)

    return SummaryWriter(log_dir=log_dir)  # de locos!!
