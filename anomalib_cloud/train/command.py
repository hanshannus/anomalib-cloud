import os
import json
import shutil
import click
from omegaconf import DictConfig
from pathlib import Path
from typing import Union
from anomalib_tools import train

INPUT_DIR = Path(os.environ.get("SM_INPUT_DIR", "."))
CHECKPOINT_DIR = Path("/opt/ml/checkpoints") if INPUT_DIR != Path() else Path()
OUTPUT_DIR = Path(os.environ.get("SM_OUTPUT_DIR", "./outputs"))
MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "./models"))


def _find_config(
    root_dir: Path = Path.cwd(),
    glob: str = "**/c*f*g*.y*ml",
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
        print(f"No config files found in {root_dir}")
        return None
    if len(configs) > 1:
        print(f"Multiple config files found in {root_dir}. Selecting first.")
        for cfg in configs:
            print(cfg)

    return configs[0].resolve()


def _resolve_path(
        path: Path,
) -> Path:
    """Interpret '~/' to '$HOME' directory and 'None' as '$CWD'.

    Parameters
    ----------
    path : Path
        Path to resolve.

    Returns
    -------
    Path
        Resolved path.
    """
    path = str(path)
    if path is None:
        path = '.'
    elif len(path) == 1 and path == "~":
        path = Path.home()
    elif len(path) == 2 and path == "~/":
        path = Path.home()
    elif len(path) > 2 and path[:2] == "~/":
        path = Path.home() / path[2:]
    else:
        path = Path(path)
    return path


def _handle_sagemaker_inputs(
        args: DictConfig,
) -> DictConfig:
    """Handle SageMaker input parameters when running on Sagemaker compute.

    Sagemaker creates so-called input channels that are provided to the training
    job as path variables through the SageMaker Python SDK Estimator class or
    the CreateTrainingJob API operation.

    Sagemaker environment variables:
    SM_INPUT_DIR = "/opt/ml/input"
    SM_CHANNELS = '["<channel name>", "<channel name>", ...]'
    SM_CHANNEL_<channel name> = "/opt/ml/input/data/<channel name>"

    Other environment variables:
    SM_OUTPUT_DIR = "/opt/ml/output"
    SM_MODEL_DIR = "/opt/ml/model"

    This function looks for the existence of SageMaker environment variables
    and adds them to the configuration dictionary. The channel path ist
    prepended to the respective argument item if that item exists.

    Parameters
    ----------
    args : DictConfig
        Configuration dictionary.

    Returns
    -------
    DictConfig
        Configuration dictionary.
    """
    if "SM_CHANNELS" not in os.environ:
        return args
    channels = json.loads(os.environ.get("SM_CHANNELS", "[]"))
    print("Sagemaker environment detected.")
    print(f"Found {len(channels)} channels.")
    for channel in channels:
        channel_dir = Path(os.environ[f"SM_CHANNEL_{channel.upper()}"])
        print(f"Path of channel '{channel}' is: {channel_dir}")
        if channel in args:
            print(f"Prepending {channel_dir} to input {args[channel]}.")
            args[channel] = channel_dir / args[channel]
        else:
            print(f"Add input {channel_dir}.")
            args[channel] = channel_dir
    return args


@click.command()
#
@click.option("--data_dir", default=Path(), type=Path)
@click.option("--config_dir", default=Path(), type=Path)
@click.option("--checkpoint_dir", default=CHECKPOINT_DIR, type=Path)
@click.option("--model_dir", default=MODEL_DIR, type=Path)
@click.option("--output_dir", default=OUTPUT_DIR, type=Path)
#
@click.option("--config", default=None, type=Path)
@click.option("--checkpoint", default=None, type=Path)
#
@click.option("--max_epochs", default=50, type=int)
@click.option("--logger", default=None, type=str)
def main(
        data_dir: Path = Path(),
        config_dir: Path = Path(),
        checkpoint_dir: Path = Path(),
        model_dir: Path = Path("./models"),
        output_dir: Path = Path("./outputs"),
        config: Path = None,
        checkpoint: Path = None,
        max_epochs: int = 50,
        logger: str = None,
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
    logger : str, optional
        Logger to use, by default None
        - mlflow: use mlflow logger
        - None: use internal logger

    Returns
    -------
    DictConfig
        Collection of parsed command line arguments.
    """
    # initialize argument dict
    args = DictConfig({})
    # resolve input directory paths ('~' -> $HOME, None -> '.')
    config_dir = _resolve_path(config_dir)
    checkpoint_dir = _resolve_path(checkpoint_dir)
    data_dir = _resolve_path(data_dir)
    output_dir = _resolve_path(output_dir)
    model_dir = _resolve_path(model_dir)
    # combine input directories with input files
    local_config = _find_config(config_dir)
    config = local_config if config is None else config_dir / config
    checkpoint = None if checkpoint is None else checkpoint_dir / checkpoint
    # add input directory paths to argument dict
    args.data_dir = data_dir
    args.output_dir = output_dir
    args.model_dir = model_dir
    args.config = config
    args.checkpoint = checkpoint
    # sagemaker default paths
    args = _handle_sagemaker_inputs(args)
    # add hyperparameters to argument dict
    args.max_epochs = max_epochs
    print(args)
    if logger == "mlflow":
        train_mlflow(args)
    else:
        train_raw(args)


def train_raw(args: DictConfig):
    print("start training")
    model_path = train(
        config_path=str(args.config),
        data_dir=str(args.data_dir),
        ckpt_path=str(args.checkpoint),
        max_epochs=args.max_epochs,
        output_dir=str(args.output_dir),
    )
    # move best model checkpoint to output dir
    print("end training")
    if args.model_dir is not None:
        shutil.move(model_path, args.model_dir / "best_model.ckpt")


def train_mlflow(args: DictConfig):
    """Run training on Azure ML.

    Parameters
    ----------
    args : DictConfig
        Input arguments.
    """
    import mlflow
    import mlflow.pytorch
    # enable auto logging
    print("MLflow automatic logging enabled")
    mlflow.pytorch.autolog()
    # log training with MLFlow
    print("MLflow run started")
    with mlflow.start_run():
        train_raw(args)


if __name__ == "__main__":
    main()
