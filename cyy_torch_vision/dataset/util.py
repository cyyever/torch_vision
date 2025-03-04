import functools
import os

import torch
import torch.utils.data
import torchvision.utils
from cyy_torch_toolbox import DatasetUtil


class VisionDatasetUtil(DatasetUtil):
    def _get_image_tensor(self, index: int) -> torch.Tensor:
        sample = self.get_sample(index)
        sample_input = sample["input"]
        if not isinstance(sample_input, torch.Tensor):
            return torchvision.transforms.ToTensor()(sample_input)
        print(type(sample_input))
        return sample_input

    @functools.cached_property
    def channel(self):
        x = self._get_image_tensor(0)
        assert x.shape[0] <= 3
        return x.shape[0]

    def get_mean_and_std(self):
        if self._name.lower() == "imagenet":
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            return (mean, std)
        mean = torch.zeros(self.channel)
        for index in range(len(self)):
            x = self._get_image_tensor(index)
            for i in range(self.channel):
                mean[i] += x[i, :, :].mean()
        mean.div_(len(self))

        wh = None
        std = torch.zeros(self.channel)
        for index in range(len(self)):
            x = self._get_image_tensor(index)
            if wh is None:
                wh = x.shape[1] * x.shape[2]
            for i in range(self.channel):
                std[i] += torch.sum((x[i, :, :] - mean[i].data.item()) ** 2) / wh
        std = std.div(len(self)).sqrt()
        return mean, std

    def save_sample_image(self, index: int, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sample_input = self._get_sample_input(index)
        if hasattr(sample_input, "save"):
            sample_input.save(path)
        else:
            torchvision.utils.save_image(sample_input, path)
