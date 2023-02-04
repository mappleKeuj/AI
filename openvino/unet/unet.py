
from pathlib import Path
import time
import numpy as np

from openvino.preprocess import PrePostProcessor
from openvino.runtime import Core, Type, Layout

model_path = Path(r"C:\Programming\computer_vision\AI\pytorch\prototypes\unet\unet.onnx")
device_name = "CPU"

# --------------------------- Step 1. Initialize OpenVINO Runtime Core ------------------------------------------------
core = Core()

# --------------------------- Step 2. Read a model --------------------------------------------------------------------
model = core.read_model(model_path)

# --------------------------- Step 3. Set up input --------------------------------------------------------------------
input_tensor = np.random.rand(1, 512, 512, 4)

# --------------------------- Step 4. Apply preprocessing -------------------------------------------------------------
ppp = PrePostProcessor(model)

_, h, w, _ = input_tensor.shape

# 1) Set input tensor information:
# - input() provides information about a single model input
# - reuse precision and shape from already available `input_tensor`
# - layout of data is 'NHWC'
ppp.input().tensor() \
    .set_shape(input_tensor.shape) \
    .set_element_type(Type.f32) \
    .set_layout(Layout('NHWC'))  # noqa: ECE001, N400

# 3) Here we suppose model has 'NCHW' layout for input
ppp.input().model().set_layout(Layout('NCHW'))

# 4) Set output tensor information:
# - precision of tensor is supposed to be 'f32'
ppp.output().tensor().set_element_type(Type.f32)

# 5) Apply preprocessing modifying the original 'model'
model = ppp.build()


# --------------------------- Step 5. Loading model to the device -----------------------------------------------------
compiled_model = core.compile_model(model, device_name)

# --------------------------- Step 6. Create infer request and do inference synchronously -----------------------------
start = time.time()
results = compiled_model.infer_new_request({0: input_tensor})
end = time.time()
print(f"time {end - start}")

# --------------------------- Step 7. Process output ------------------------------------------------------------------
predictions = next(iter(results.values()))

a = 0