# 🧠 Deep Learning Task – YOLOv11

## 3️⃣ Model Conversion

The **YOLOv11** model (saved as a `.pt` file) was converted into a **deployment-ready ONNX format** for inference on the **NVIDIA Triton Inference Server**.

All conversion logic is implemented in the [`model_conversion.py`](model_conversion.py) script.

### 🔄 Conversion Details

The conversion process uses the built-in `model.export()` method from the **Ultralytics YOLO** library.  
The trained model `best.pt` was exported to **ONNX** format with the following key parameters:

| Parameter | Description |
|------------|--------------|
| `simplify=True` | Simplifies the ONNX computation graph to remove redundant operations and improve inference speed. |
| `nms=True` | Integrates **Non-Max Suppression (NMS)** directly into the ONNX graph, making the exported model fully end-to-end (includes post-processing). |

After export, the resulting ONNX model is stored in a **Triton-compatible repository structure**:

```
├── model_repository/
│   └── yolo11/
│       ├── config.pbtxt        # Triton configuration file
│       └── 1/
│           └── model.onnx      # Exported ONNX model
```


## 4️⃣ Model Deployment

The deployment of the YOLOv11 ONNX model was performed using the **NVIDIA Triton Inference Server**, running inside a **Docker container**.  

It is implemented in the [`model_deployment.py`](model_deployment.py) script which performs inference using both **PyTorch** and **Triton**, compares their outputs, and saves the visualization and error analysis results.

---

### ⚙️ Deployment Steps

1. **Logging in to NVIDIA NGC Registry**
   Before pulling the Triton image, we authenticate with our NVIDIA NGC account:
   ```bash
   docker login nvcr.io
   Username: $oauthtoken
   Password: <your NGC API key>
   ```

2. **Pulling the Triton Inference Server Image**
   ```bash
   docker pull nvcr.io/nvidia/tritonserver:24.07-py3
   ```

3. **Preparing the Model Repository**
   We make sure our model repository is available locally with the following structure:
   ```
   model_repository/
   └── yolo11/
       ├── config.pbtxt
       └── 1/
           └── model.onnx
   ```

4. **Running Triton Server**
   Starting the server inside a Docker container by mounting our local repository:
   ```bash
   docker run -d \
     --name triton_yolo11 \
     -p 8000:8000 -p 8001:8001 -p 8002:8002 \
     -v /home/fidan/model_repository:/models \
     nvcr.io/nvidia/tritonserver:24.07-py3 \
     tritonserver --model-repository=/models
   ``'

---

### 🧠 Inference and Validation

After deployment, inference was performed using the **Triton Python HTTP client**.  
The [`model_deployment.py`](model_deployment.py) script runs the following process:

- Loads a sample test image.  
- Sends it to the Triton server for inference via the `httpclient.InferenceServerClient`.  
- Runs the same inference locally using the PyTorch model (`best.pt`).  
- Compares the predicted bounding boxes, confidence scores, and keypoints.  
- Saves visualization outputs as well as error metrics in:
  ```
  deployment_results/
  ```

---
---


