import functools

from cyy_torch_toolbox import DatasetType
from cyy_torch_toolbox.dataset.collection import DatasetCollection
from cyy_torch_toolbox.factory import Factory
from cyy_torch_toolbox.model import (create_model,
                                     global_model_evaluator_factory,
                                     global_model_factory)
from cyy_torch_toolbox.model.repositary import get_model_info

from .evaluator import VisionModelEvaluator


def get_model_evaluator(model, **kwargs):
    return VisionModelEvaluator(model=model, **kwargs)


global_model_evaluator_factory.register(DatasetType.Vision, VisionModelEvaluator)


def get_model(
    model_constructor_info: dict, dataset_collection: DatasetCollection, **kwargs
) -> dict:
    final_model_kwargs: dict = kwargs
    dataset_util = dataset_collection.get_dataset_util()
    for k in ("input_channels", "channels"):
        if k not in kwargs:
            final_model_kwargs |= {
                k: dataset_util.channel,
            }
    model = create_model(model_constructor_info["constructor"], **final_model_kwargs)
    return {"model": model, "repo": model_constructor_info.get("repo", None)}


model_constructors = get_model_info().get(DatasetType.Vision, {})
for name, model_constructor_info in model_constructors.items():
    if DatasetType.Vision not in global_model_factory:
        global_model_factory[DatasetType.Vision] = Factory()
    global_model_factory[DatasetType.Vision].register(
        name, functools.partial(get_model, model_constructor_info)
    )

global_model_evaluator_factory.register(DatasetType.Vision, get_model_evaluator)
