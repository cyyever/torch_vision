import torch
import torch.utils.data
import torchvision.transforms
from cyy_naive_lib.log import log_debug
from cyy_torch_toolbox import (
    DatasetCollection,
    DatasetType,
    MachineLearningPhase,
    TransformType,
)
from cyy_torch_toolbox.data_pipeline.transform import (
    DataPipeline,
    Transform,
    Transforms,
)

from ..dataset.util import VisionDatasetUtil


def get_mean_and_std(dc):
    dataset = torch.utils.data.ConcatDataset(list(dc.foreach_dataset()))
    pipeline = DataPipeline()
    pipeline.append(Transform(fun=torchvision.transforms.ToTensor()))

    def computation_fun():
        return VisionDatasetUtil(
            dataset=dataset,
            transforms=Transforms(),
            pipeline=pipeline,
            name=dc.name,
        ).get_mean_and_std()

    return dc.get_cached_data("mean_and_std.pk", computation_fun)


def add_vision_extraction(dc: DatasetCollection) -> None:
    assert dc.dataset_type == DatasetType.Vision
    dc.append_transform(torchvision.transforms.ToTensor(), key=TransformType.Input)
    dc.append_named_transform(
        Transform(
            name="to_tensor", fun=torchvision.transforms.ToTensor(), cacheable=True
        ),
    )


def add_vision_transforms(dc: DatasetCollection, model_evaluator) -> None:
    assert dc.dataset_type == DatasetType.Vision
    mean, std = get_mean_and_std(dc)
    dc.append_transform(
        torchvision.transforms.Normalize(mean=mean, std=std),
        key=TransformType.Input,
    )
    dc.append_named_transform(
        Transform(
            name="normalize",
            fun=torchvision.transforms.Normalize(mean=mean, std=std),
            cacheable=True,
        ),
    )
    input_size = getattr(
        model_evaluator.get_underlying_model().__class__, "input_size", None
    )
    if input_size is not None:
        log_debug("resize input to %s", input_size)
        dc.append_transform(
            transform=torchvision.transforms.Resize(input_size, antialias=True),
            key=TransformType.Input,
        )
        dc.append_named_transform(
            Transform(
                fun=torchvision.transforms.Resize(input_size, antialias=True),
                cacheable=True,
            )
        )
    if dc.name.upper() not in ("SVHN", "MNIST"):
        dc.append_transform(
            torchvision.transforms.RandomHorizontalFlip(),
            key=TransformType.RandomInput,
            phases={MachineLearningPhase.Training},
        )
        dc.append_named_transform(
            Transform(
                fun=torchvision.transforms.RandomHorizontalFlip(),
            ),
            phases={MachineLearningPhase.Training},
        )
    if dc.name.upper() in ("CIFAR10", "CIFAR100"):
        dc.append_transform(
            torchvision.transforms.RandomCrop(32, padding=4),
            key=TransformType.RandomInput,
            phases={MachineLearningPhase.Training},
        )
        dc.append_named_transform(
            Transform(
                fun=torchvision.transforms.RandomCrop(32, padding=4),
            ),
            phases={MachineLearningPhase.Training},
        )
