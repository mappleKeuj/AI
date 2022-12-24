
import torch 
import logging

from detection import VGG
from detection.config import VggConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

logging.info(f"Use device: {device}")

img = torch.randn([1, 3, 300, 300]).to(device)

config_name = VggConfig().get_list_of_configs()[0]
vgg_config = VggConfig(config_name=config_name, use_batch_norm=True)
logging.info(f"Create vgg with config: {repr(vgg_config)}")
vgg_model = VGG(vgg_config, in_channels=3)


print(vgg_model)

vgg_model = vgg_model.to(device)
output = vgg_model(img)



# import torch
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11_bn', pretrained=True)
# model

# print(model)