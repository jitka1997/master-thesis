# pylint: disable=missing-docstring
import torchvision
import torch

resnet = torchvision.models.resnet50(pretrained=True)

# Toto bude dobra matica (alebo akakolvek ina layer)
W = resnet.layer4[1].conv2.weight.flatten(1)
W.shape

# Ulozi sa ako
torch.save(W, "xy.pt")

# Loadne sa ako
# W2 = torch.load("xy.pt").detach().numpy()
