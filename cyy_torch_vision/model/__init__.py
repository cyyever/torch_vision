import functools

from cyy_torch_toolbox import DatasetCollection, DatasetType, Factory
from cyy_torch_toolbox.model import (
    create_model,
    global_model_evaluator_factory,
    global_model_factory,
)
from cyy_torch_toolbox.model.repositary import get_model_info, get_torch_hub_model_info

from ..dataset.util import VisionDatasetUtil
from .evaluator import VisionModelEvaluator

global_model_evaluator_factory.register(DatasetType.Vision, [VisionModelEvaluator])


def __get_model(
    model_constructor_info: dict, dataset_collection: DatasetCollection, **kwargs
) -> dict:
    final_model_kwargs: dict = kwargs
    dataset_util = dataset_collection.get_dataset_util()
    assert isinstance(dataset_util, VisionDatasetUtil)
    for k in ("input_channels", "channels"):
        if k not in kwargs:
            final_model_kwargs |= {
                k: dataset_util.channel,
            }
    model = create_model(model_constructor_info["constructor"], **final_model_kwargs)
    return {"model": model, "repo": model_constructor_info.get("repo")}


def __get_model_constructors() -> dict:
    model_info: dict = {}
    github_repos: list = [
        "huggingface/pytorch-image-models:main",
        "pytorch/vision:main",
    ]

    for repo in github_repos:
        model_info |= get_torch_hub_model_info(repo)
    model_info |= get_model_info()[DatasetType.Vision]
    return model_info


__factory = Factory()
for name, constructor_info in __get_model_constructors().items():
    __factory.register(name, functools.partial(__get_model, constructor_info))

if DatasetType.Vision not in global_model_factory:
    global_model_factory[DatasetType.Vision] = []

global_model_factory[DatasetType.Vision].append(__factory)
