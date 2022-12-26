
import torch

from ssd.model import SSD300

ssd_model = SSD300(backbone="resnet18")
out = ssd_model(torch.randn([16,3,300,300]))
print(out[0].shape)
print(out[1].shape)
