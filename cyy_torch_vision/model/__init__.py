import functools

import torch
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox import DatasetType
from cyy_torch_toolbox.dataset.collection import DatasetCollection
from cyy_torch_toolbox.factory import Factory
from cyy_torch_toolbox.model import (create_model,
                                     global_model_evaluator_factory,
                                     global_model_factory)

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


def get_model_constructors() -> dict:
    model_info: dict = {}
    github_repos: list = [
        "pytorch/vision:main",
        "huggingface/pytorch-image-models:main",
        "cyyever/torch_models:main",
    ]

    for repo in github_repos:
        entrypoints = torch.hub.list(
            repo, force_reload=False, trust_repo=True, skip_validation=True
        )
        for model_name in entrypoints:
            if model_name.lower() not in model_info:
                model_info[model_name.lower()] = {
                    "name": model_name,
                    "constructor": functools.partial(
                        torch.hub.load,
                        repo_or_dir=repo,
                        model=model_name,
                        force_reload=False,
                        trust_repo=True,
                        skip_validation=True,
                        verbose=False,
                    ),
                    "repo": repo,
                }
            else:
                get_logger().debug("ignore model_name %s", model_name)

    return model_info


if DatasetType.Vision not in global_model_factory:
    global_model_factory[DatasetType.Vision] = Factory()
for name, constructor_info in get_model_constructors().items():
    global_model_factory[DatasetType.Vision].register(
        name, functools.partial(get_model, constructor_info)
    )

global_model_evaluator_factory.register(DatasetType.Vision, get_model_evaluator)
