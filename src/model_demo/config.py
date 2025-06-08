from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PathConfigSchema:
    """
    Configuration schema for the metadata.

    Parameters
    ----------
    data_dir : str
        the directory where data is saved.
    data_fname : str
        the name of data file.
    model_dir : str
        the directory where models are saved.
    model_fname: str
        the name of the model .
    """

    data_dir: str = "data/model_demo"
    data_fname: str = "data_tensors.pt"
    model_dir: str = "models/model_demo"
    model_fname: str = "demo_model_weights.pth"

@dataclass
class ModelParametersConfigSchema:
    """
    Configuration schema for the model training parameters.

    Parameters
    ----------

    """
    train_size: float = field(default=0.8)
    batch_size: int = field(default=100)   # batch_size should be a positive integer value
    epochs: int = field(default=100)
    learning_rate: float = field(default=0.01)


#@dataclass
class MetadataConfigSchema:
    """
    Configuration schema for the full training workflow.

    Parameters
    ----------
    data : DataConfigSchema
        Schema defining the data
    model : ModelConfigSchema
        Schema defining the model
    output_dir : str
        directory in which to store the trained model
    """

    data: PathConfigSchema = PathConfigSchema()
    model: ModelParametersConfigSchema = ModelParametersConfigSchema()
    # output_dir: str = ("")
