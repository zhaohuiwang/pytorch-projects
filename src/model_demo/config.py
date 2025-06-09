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

    data_dir: str = field(default="data/model_demo")
    data_fname: str = field(default="data_tensors.pt")
    model_dir: str = field(default="models/model_demo")
    model_fname: str = field(default="demo_model_weights.pth")

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
    Configuration schema for the full training workflow - classes grouping.

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


""""
@dataclass decorator is implementing __init__(constructor), (object string representation), __eq__( equality operator) classes behind the scenes
Data classes require type hints but types aren't actually enforced due to Python not dataclass itself. Data classes also allow default values in fields. Keep in mind that non-default fields can’t follow default fields.

In practice, you will rarely define defaults with name: type = value syntax. Instead, you will use the **field function**, which allows more control of each field definition. Syntax: 
name: type = field(default=value)
The default_factory parameter accepts a function that returns an initial value for a data class field. It accepts any arbitratary funciton, including tuple, list, dict, set, and any user-defined custum function or lambda <arguments>  : expression
for example, exercises: List[Exercise] = field(default_factory=create_warmup)
We can add methods to data classes as we do for regular classes.



"""