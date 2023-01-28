
import torch
import time
from unet import UnetGenerator


unet = UnetGenerator(output_nc=4, input_nc=4, num_downs=9)
unet.eval()
unet = unet.to("cuda:0")
print(unet)

dummy = torch.rand(1, 4, 512, 512)
dummy = dummy.to("cuda:0")
print(dummy)
start = time.time()
out = unet(dummy)
end = time.time()

print(f"Process time {end - start}")


print(out.shape)

torch.onnx.export(unet, dummy, "unet.onnx", export_params=True)
