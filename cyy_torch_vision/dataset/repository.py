import torch
import torch.utils.data
import torchvision.datasets
from cyy_naive_lib.reflection import get_class_attrs
from cyy_torch_toolbox import DatasetType
from cyy_torch_toolbox.dataset.repository import register_dataset_constructors


def register_constructors() -> None:
    repositories = [
        torchvision.datasets,
    ]
    dataset_constructors: dict = {}
    for repository in repositories:
        dataset_constructors |= get_class_attrs(
            repository,
            filter_fun=lambda _, v: issubclass(v, torch.utils.data.Dataset),
        )

    for name, constructor in dataset_constructors.items():
        register_dataset_constructors(
            name=name, constructor=constructor, dataset_type=DatasetType.Vision
        )
