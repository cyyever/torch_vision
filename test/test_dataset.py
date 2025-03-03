from cyy_torch_toolbox import (
    ClassificationDatasetCollection,
    MachineLearningPhase,
    create_dataset_collection,
)
from cyy_torch_vision import VisionDatasetUtil


def test_dataset() -> None:
    mnist = create_dataset_collection("MNIST")
    assert (
        abs(
            len(mnist.get_dataset_util(MachineLearningPhase.Training))
            / len(mnist.get_dataset_util(MachineLearningPhase.Validation))
            - 12
        )
        < 0.01
    )
    dataset_util = mnist.get_dataset_util()
    assert isinstance(dataset_util, VisionDatasetUtil)
    assert dataset_util.channel == 1

    cifar10 = create_dataset_collection("CIFAR10")
    assert isinstance(cifar10, ClassificationDatasetCollection)
    dataset_util = cifar10.get_dataset_util()
    assert isinstance(dataset_util, VisionDatasetUtil)
    assert dataset_util.channel == 3
    assert len(dataset_util.get_labels()) == 10
    assert abs(
        len(cifar10.get_dataset_util(MachineLearningPhase.Test))
        - len(cifar10.get_dataset_util(MachineLearningPhase.Validation))
        <= 1
    )
    print("cifar10 labels are", cifar10.get_label_names())


def test_dataset_labels() -> None:
    for name in ("MNIST", "CIFAR10"):
        dc = create_dataset_collection(name)
        assert isinstance(dc, ClassificationDatasetCollection)
        assert len(dc.get_labels()) == 10
