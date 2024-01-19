import pathlib
from typing import Dict, Any
import torch


class Checkpointer:
    def __init__(self):
        pass

    @staticmethod
    def checkpoint(checkpoint_dict: Dict[str, Any], path: pathlib.Path):
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            ...
            }, PATH)

        Parameters
        ----------
        checkpoint_dict: Dict[str, Any]
            The key-value dictionary as shown above to save
        path: pathlib.Path
            The filename to store the checkpiont

        Returns
        -------
        None
        """
        torch.save(checkpoint_dict, str(path))
