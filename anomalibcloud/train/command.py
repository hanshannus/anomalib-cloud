import os
import json
import shutil
import click
from loguru import logger
from omegaconf import DictConfig
from pathlib import Path
from typing import Union, Dict
from anomalibtools import train
from anomalib.config import get_configurable_parameters

INPUT_DIR = Path(os.environ.get("SM_INPUT_DIR", "."))
CHECKPOINT_DIR = Path("/opt/ml/checkpoints") if INPUT_DIR != Path() else Path()
OUTPUT_DIR = Path(os.environ.get("SM_OUTPUT_DIR", "./outputs"))
MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "./models"))


def _find_config(
    root_dir: Path = Path.cwd(),
    glob: str = "**/config*.y*ml",
) -> Union[None, Path]:
    """Search for configuration YAML file.

    Parameters
    ----------
    glob : str, optional
        Search pattern, by default "**/c*f*g*.y*ml"

    Returns
    -------
    Path
        Path to configuration YAML file.

    Raises
    ------
    FileNotFoundError
        Configuration file not found.
    """
    configs = [i for i in root_dir.glob(glob)]

    if len(configs) == 0:
        raise FileNotFoundError(
            f"No config files found in {root_dir}. The name of the config file"
            f"must match the glob pattern '{glob}' to be detected."
        )
    if len(configs) > 1:
        logger.warning(f"Multiple config files found in {root_dir}. Selecting first.")
        for cfg in configs:
            print(cfg)

    return configs[0].resolve()


def _resolve_path(
    path: Union[str, Path],
) -> Path:
    """Interpret '~/' to '$HOME' directory and 'None' as '$CWD'.

    Parameters
    ----------
    path : Path
        Path to resolve.

    Returns
    -------
    Path
        Resolved directory_path.
    """
    path = str(path)
    if path is None:
        path = "."
    elif len(path) == 1 and path == "~":
        path = Path.home()
    elif len(path) == 2 and path == "~/":
        path = Path.home()
    elif len(path) > 2 and path[:2] == "~/":
        path = Path.home() / path[2:]
    else:
        path = Path(path)
    return path


def _resolve_or_replace_by_sagemaker_channel(
    directory_path: Path,
    channel_name: str = None,
) -> Path:
    """Resolve or replace the input directory_path by SageMaker channel.

    Sagemaker environment variables:
    SM_INPUT_DIR = "/opt/ml/input"
    SM_CHANNELS = '["<channel name>", "<channel name>", ...]'
    SM_CHANNEL_<channel name> = "/opt/ml/input/data/<channel name>"

    Parameters
    ----------
    directory_path : Path
        Input directory_path.
    channel_name : str, optional
        Channel name, by default None.

    Returns
    -------
    Path
        Resolved directory_path.
    """
    env = os.environ
    if channel_name is None:
        return _resolve_path(directory_path)
    # detect sagemaker input channels
    channels = json.loads(env.get("SM_CHANNELS", "[]"))
    # check if channel name exists in environment variables
    channel_name = channel_name.upper()
    if channel_name in channels:
        return Path(env[f"SM_CHANNEL_{channel_name}"])
    # resolve input directory paths ('~' -> $HOME, None -> '.')
    return _resolve_path(directory_path)


@click.command()
# SageMaker channel input directories (defined in 'Estimator.fit')
@click.option("--data_dir", default=Path(), type=Path)
@click.option("--config_dir", default=Path(), type=Path)
# local input directories
@click.option("--checkpoint_dir", default=CHECKPOINT_DIR, type=Path)
@click.option("--model_dir", default=MODEL_DIR, type=Path)
@click.option("--output_dir", default=OUTPUT_DIR, type=Path)
# local paths
@click.option("--config", default=None, type=Path)
@click.option("--checkpoint", default=None, type=Path)
# - hyperparameters
@click.option("--max_epochs", default=50, type=int)
@click.option("--monitor", default=None, type=str)
def main(
    data_dir: Path = Path(),
    config_dir: Path = Path(),
    checkpoint_dir: Path = Path(),
    model_dir: Path = Path("./models"),
    output_dir: Path = Path("./outputs"),
    config: Path = None,
    checkpoint: Path = None,
    max_epochs: int = 50,
    monitor: str = None,
):
    """Parse command line arguments.

    Parameters
    ----------
    data_dir : Path, optional
        Path to data directory.
    config_dir : Path, optional
        Path to config directory.
    checkpoint_dir : Path, optional
        Path to checkpoint directory.
    config : Path, optional
        Path to config file, by default None
        If None, search for config file in current working directory.
    checkpoint : Path, optional
        Path to model checkpoint file, by default None
    model_dir : Path, optional
        Output directory for model training results, by default None
    max_epochs : int, optional
        Maximum number of training epochs, by default 50
    output_dir : Path, optional
        Output directory for outputs, by default None
    monitor : str, optional
        Logger to use, by default None
        - mlflow: use mlflow logger
        - None: use internal logger

    Returns
    -------
    DictConfig
        Collection of parsed command line arguments.
    """
    # resolve input directory paths
    config_dir = _resolve_or_replace_by_sagemaker_channel(config_dir, "config_dir")
    data_dir = _resolve_or_replace_by_sagemaker_channel(data_dir, "data_dir")
    checkpoint_dir = _resolve_or_replace_by_sagemaker_channel(checkpoint_dir)
    output_dir = _resolve_or_replace_by_sagemaker_channel(output_dir)
    model_dir = _resolve_or_replace_by_sagemaker_channel(model_dir)
    # combine input directories with input files
    config = _find_config(config_dir) if config is None else config_dir / config
    checkpoint = None if checkpoint is None else checkpoint_dir / checkpoint
    # load configuration
    cfg = get_configurable_parameters(config_path=str(config))
    # update paths
    cfg["dataset"]["path"] = str(data_dir)
    cfg["project"]["path"] = str(output_dir)
    # cfg["trainer"]["ckpt_path"] = str(checkpoint)
    # hyperparameters
    cfg["trainer"]["max_epochs"] = max_epochs
    #
    if monitor == "mlflow":
        train_mlflow(cfg, model_dir=model_dir)
    else:
        train_raw(cfg, model_dir=model_dir)


def train_raw(config: DictConfig, model_dir: Path = None):
    """Run training.

    Parameters
    ----------
    config : DictConfig
        Configuration.
    model_dir : Path, optional
        Output directory for best model checkpoint, by default None
    """
    logger.info("start training")
    trainer = train(config=config)
    # move best model checkpoint to output dir
    logger.info("end training")
    if model_dir is not None:
        model_path = trainer.checkpoint_callback.best_model_path
        shutil.copyfile(model_path, model_dir / "best_model.ckpt")


def train_mlflow(config: DictConfig, model_dir: Path = None):
    """Run training with MFlow.

    Parameters
    ----------
    config : DictConfig
        Configuration.
    model_dir : Path, optional
        Output directory for best model checkpoint, by default None
    """
    import mlflow
    import mlflow.pytorch

    # enable auto logging
    logger.info("MLFlow automatic logging enabled")
    mlflow.pytorch.autolog()
    # log training with MLFlow
    logger.info("MLFlow run started")
    with mlflow.start_run():
        train_raw(config, model_dir=model_dir)


if __name__ == "__main__":
    main()
