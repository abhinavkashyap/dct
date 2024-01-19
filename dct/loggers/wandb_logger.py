import wandb
from typing import Dict, Any, List, Union, Tuple
import pathlib


class WeightsAndBiasesLogger:
    """
    A really light-weight wrapper around WandB loggiting
    """

    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        config: Dict[str, Any] = None,
    ):
        self.project_name = project_name

        self.config = config if config is not None else {}

        # initialize the config with an empty config
        wandb.init(config=self.config, project=project_name, name=experiment_name)

    @staticmethod
    def save_hyperparams(hyper_params: Dict[str, Any]):
        """Saves hyper-parameters to Weights and Biases

        Parameters
        ----------
        hyper_params: Dict[str, Any]
            Dictionary of hyper-parameters

        Returns
        -------
        None

        """
        wandb.config.update(hyper_params)

    @staticmethod
    def log(data: Dict[str, Any], step=None):
        """Log something to wandb

        Parameters
        ----------
        data: Dict[str, Any]
            Things to log
        step: int
            The step number - Corresponds to one forward pass in the neural network

        Returns
        -------
        None
        """
        wandb.log(data, step=step)

    @staticmethod
    def log_table(
        table_name: str,
        table_data: Union[List[List[str]], List[Tuple[str, ...]]],
        columns: List[str],
    ):
        wandb.log({table_name: wandb.Table(data=table_data, columns=columns)})

    @staticmethod
    def log_image(image_name: str, path: Union[str, pathlib.Path]):
        wandb.Image({image_name: wandb.Image(str(path))})
