from cyy_torch_toolbox import DatasetType
from cyy_torch_toolbox.dataset.util import global_dataset_util_factor

from .repository import register_constructors
from .util import VisionDatasetUtil

global_dataset_util_factor.register(DatasetType.Vision, VisionDatasetUtil)
register_constructors()
