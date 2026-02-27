import functools
from typing import Any

from cyy_torch_toolbox import DatasetCollection, DatasetType, Factory
from cyy_torch_toolbox.model import (
    create_model,
    global_model_factory,
)

from ..dataset.util import VisionDatasetUtil


def _get_model(
    model_constructor_info: dict[str, Any],
    dataset_collection: DatasetCollection,
    **kwargs: Any,
) -> dict[str, Any]:
    final_model_kwargs: dict[str, Any] = kwargs
    dataset_util = dataset_collection.get_any_dataset_util()
    assert isinstance(dataset_util, VisionDatasetUtil)
    for k in ("input_channels", "channels"):
        if k not in kwargs:
            final_model_kwargs |= {
                k: dataset_util.channel,
            }
    model = create_model(model_constructor_info["constructor"], **final_model_kwargs)
    return {"model": model, "repo": model_constructor_info.get("repo")}


class _LazyModelFactory(Factory):
    """Lazily loads torch hub models on first access."""

    def __init__(self) -> None:
        super().__init__()
        self.__loaded = False

    def __load(self) -> None:
        if self.__loaded:
            return
        self.__loaded = True
        from cyy_torch_toolbox.model.repository import get_model_info

        for name, constructor_info in get_model_info()[DatasetType.Vision].items():
            self.register(name, functools.partial(_get_model, constructor_info))

    def get(
        self, key: Any, case_sensitive: bool = True, default: Any = None, **kwargs: Any
    ) -> Any:
        self.__load()
        return super().get(key, case_sensitive=case_sensitive, default=default, **kwargs)

    def get_similar_keys(self, key: str) -> list[str]:
        self.__load()
        return super().get_similar_keys(key)


if DatasetType.Vision not in global_model_factory:
    global_model_factory[DatasetType.Vision] = []

global_model_factory[DatasetType.Vision].append(_LazyModelFactory())
