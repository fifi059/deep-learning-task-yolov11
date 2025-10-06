import os
import cv2
import numpy as np
import glob
from ultralytics import YOLO
import tritonclient.http as httpclient


# -------------------------
# CONFIGURATION
# -------------------------
IMG_SIZE = 640
CONF_THRESH = 0.25

TRITON_URL = "localhost:8000"
print(f"Connecting to Triton server at {TRITON_URL}")
triton_client = httpclient.InferenceServerClient(url=TRITON_URL)

# -------------------------
# PATHS
# -------------------------
TEST_IMAGES_PATH = "/mnt/g/My Drive/YOLO_dataset/images/test/*.jpg"
LABELS_DIR       = "/mnt/g/My Drive/YOLO_dataset/labels/test"
PT_WEIGHT_PATH    = "/mnt/g/My Drive/YOLO_dataset/results/train4/weights/best.pt"

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# FUNCTIONS
# -------------------------
def letterboxing(image, new_shape=640):
    
    # Getting original h,w and finding scaling ratio
    shape = image.shape[:2]
    scaling_ratio = min(new_shape / shape[0], new_shape / shape[1])
    
    # New resized size before padding
    new_unpadded = (int(round(shape[1] * scaling_ratio)), int(round(shape[0] * scaling_ratio)))
    
    # Calculating empty space left, dividing by 2 to pad evenly
    dw, dh = new_shape - new_unpadded[0], new_shape - new_unpadded[1]
    dw/=2
    dh/=2
    
    # Resizing Image
    image_resized = cv2.resize(image, new_unpadded, interpolation=cv2.INTER_LINEAR)
    
    # Adding paddings
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    
    image_padded = cv2.copyMakeBorder(image_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    return image_padded, scaling_ratio, (dw, dh)
    
    
def triton_inference(image):
    
    image = image.astype(np.float32)
    
    input = httpclient.InferInput("images", image.shape, "FP32")
    input.set_data_from_numpy(image)
    
    output = httpclient.InferRequestedOutput("output0")
    
    # sending request to Triton
    result = triton_client.infer("yolo11", [input], outputs=[output])
    
    return result.as_numpy("output0")

def load_gt(label_file):
    
    label = np.loadtxt(label_file).reshape(-1)
    
    bbox_labels = label[1:5]
    keypoint_labels = label[5:].reshape(-1, 2)
    
    return bbox_labels, keypoint_labels

def denorm_bbox(gt_bbox, image_shape):
    
    h,w = image_shape[:2]
    x, y, bw, bh = gt_bbox
    
    return np.array([x*w, y*h, bw*w, bh*h], dtype=np.float32)

def denorm_kpts(gt_kpts, image_shape):
    
    h,w = image_shape[:2]
    return np.stack([gt_kpts[:, 0] * w, gt_kpts[:, 1] * h], axis=1).astype(np.float32)

def bbox_error(pred_bbox, gt_bbox):
    
    return float(np.linalg.norm(pred_bbox - gt_bbox))

def kpt_error(pred_kpts, gt_kpts):
    
    n = min(len(pred_kpts), len(gt_kpts))
    return float(np.mean(np.linalg.norm(pred_kpts[:n] - gt_kpts[:n], axis=1)))
    
def draw_bbox_kpts(image, bbox, kpts, color, label):
    
    # drawing bboxs with corresponding color and label
    x, y, w, h = bbox
    top_left = (int(x - w/2), int(y - h/2))
    bottom_right = (int(x + w/2), int(y + h/2))
    
    cv2.rectangle(image, top_left, bottom_right, color, 2)
    cv2.putText(image, label, (top_left[0], top_left[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # drawing keypoints
    if kpts is not None:
        for (kx, ky) in kpts:
            cv2.circle(image, (int(kx), int(ky)), 3, color, -1)
    
    

# --------------------------
# LOAD MODEL
# --------------------------
print("Checking model path:", PT_WEIGHT_PATH, "Exists:", os.path.exists(PT_WEIGHT_PATH))
pt_model = YOLO(PT_WEIGHT_PATH)
image_paths = sorted(glob.glob(TEST_IMAGES_PATH))
print(f"Found {len(image_paths)} test images.")

# --------------------------
# MAIN LOOP
# --------------------------

# considering first 2 images
for img_path in image_paths[:2]:
    
    filename = os.path.basename(img_path)
    print(f"\n --- Processing: {filename} ---")
    
    label_file = os.path.join(LABELS_DIR, filename.replace(".jpg", ".txt"))
    if not os.path.exists(label_file):
        print(f"Ground truth is not found at : {label_file}")
        continue    
    
    sample_image = cv2.imread(img_path)
    if sample_image is None:
        print(f"Failed to read image from: {img_path}")
        continue
    
    # Denormalizing ground truth
    gt_bbox, gt_kpts = load_gt(label_file)
    gt_bbox_denorm = denorm_bbox(gt_bbox, sample_image.shape)
    gt_kpts_denorm = denorm_kpts(gt_kpts, sample_image.shape)
    
    
    # ****************************************
    # Running PyTorch inference
    # ****************************************
    pt_output = pt_model(img_path, imgsz=IMG_SIZE, conf=0.25)
    
    pt_bbox = pt_output[0].boxes.xywh.cpu().numpy()
    pt_conf = pt_output[0].boxes.conf.cpu().numpy()
    pt_kpts = pt_output[0].keypoints.xy.cpu().numpy()
    
    # ****************************************
    # Running Triton inference
    # ****************************************
    image0, ratio, dwdh = letterboxing(sample_image, new_shape=IMG_SIZE)
    image0 = image0.astype(np.float32) / 255.0
    image0 = np.transpose(image0, (2, 0, 1))[None, ...]  
    
    triton_output = triton_inference(image0)
    print("Triton output shape:", triton_output.shape)
    triton_output = triton_output[0]
    
    conf = triton_output[:, 4]
    mask = conf > CONF_THRESH
    triton_output = triton_output[mask]
    
    # picking higest confidence detection
    best_idx = int(np.argmax(triton_output[:, 4]))
    triton_output = triton_output[best_idx]
    
    x1, y1, x2, y2 = triton_output[0:4]
    triton_bbox = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, (x2 - x1), (y2 - y1)], dtype=np.float32)
    
    triton_conf = float(triton_output[4])
    triton_cls = int(triton_output[5])
    triton_kpts = triton_output[6:]
    N = len(triton_kpts) // 2
    triton_kpts = triton_kpts.reshape(N, 2).astype(np.float32)
    
    # bbox : taking back to image coordinates
    x, y, w, h = triton_bbox
    x = (x - dwdh[0]) / ratio
    y = (y - dwdh[1]) / ratio
    w = w / ratio
    h = h / ratio
    triton_bbox = np.array([x, y, w, h], dtype=np.float32)
    
    # keypoints : taking back to image coordinates
    triton_kpts = triton_kpts.copy()
    triton_kpts[:, 0] = (triton_kpts[:, 0] - dwdh[0]) / ratio
    triton_kpts[:, 1] = (triton_kpts[:, 1] - dwdh[1]) / ratio
    
    # ***********************************
    # COMPARISONS
    # ***********************************
    
    # Computing errors wrt Ground-truth
    bbox_error_pt = bbox_error(pt_bbox[0], gt_bbox_denorm)
    bbox_error_triton = bbox_error(triton_bbox, gt_bbox_denorm)
    
    kpt_error_pt = kpt_error(pt_kpts[0], gt_kpts_denorm)
    kpt_error_triton = kpt_error(triton_kpts, gt_kpts_denorm)
    
    # Computing differences between PyTorch and Triton outputs
    bbox_diff = np.abs(pt_bbox[0] - triton_bbox)
    bbox_diff_mean = float(np.mean(bbox_diff))

    kpt_diff = np.abs(pt_kpts[0] - triton_kpts)
    kpt_diff_mean = float(np.mean(kpt_diff))
    
    # ***********************************
    # Visualizing & Saving results
    # ***********************************
    visualize_gt = sample_image.copy()
    visualize_pt = sample_image.copy()
    visualize_triton = sample_image.copy()
    
    draw_bbox_kpts(visualize_gt, gt_bbox_denorm, gt_kpts_denorm, (0, 255, 0), "GT")
    draw_bbox_kpts(visualize_pt, pt_bbox[0], pt_kpts[0], (255, 0, 0), "PT")
    draw_bbox_kpts(visualize_triton, triton_bbox, triton_kpts, (0, 0, 255), f"Triton") 
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename.replace(".jpg", "_gt.jpg")), visualize_gt)
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename.replace(".jpg", "_pt.jpg")), visualize_pt)
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename.replace(".jpg", "_tr.jpg")), visualize_triton)    
    
    txt_path = os.path.join(OUTPUT_DIR, filename.replace(".jpg", "_results.txt"))   
    with open(txt_path, "w") as f:
        # Error wrt ground-truth
        f.write(f"Pytorch vs Ground-truth BBox Error: {bbox_error_pt:.3f}, Pytorch Kpt Error: {kpt_error_pt:.3f}\n")
        f.write(f"Triton vs Ground-truth  BBox Error: {bbox_error_triton:.3f}, Triton  Kpt Error: {kpt_error_triton:.3f}\n\n")

        # Cross-model comparison
        f.write(f"PyTorch - Triton BBox error (mean): {bbox_diff_mean:.3f}\n")
        f.write(f"PyTorch - Triton Kpt  error (mean): {kpt_diff_mean:.3f}\n\n")

        # Rounding before saving
        gt_bbox_rounded = np.round(gt_bbox_denorm.astype(np.float64), 1).tolist()
        pt_bbox_rounded = np.round(pt_bbox[0].astype(np.float64), 1).tolist()
        triton_bbox_rounded = np.round(triton_bbox.astype(np.float64), 1).tolist()

        gt_kpts_rounded = np.round(gt_kpts_denorm.astype(np.float64), 1).tolist()
        pt_kpts_rounded = np.round(pt_kpts[0].astype(np.float64), 1).tolist()
        triton_kpts_rounded = np.round(triton_kpts.astype(np.float64), 1).tolist()

        # Bbox and kpts ground-truth and predictions
        f.write(f"Ground Truth BBox: {gt_bbox_rounded}\n")
        f.write(f"Pytorch BBox: {pt_bbox_rounded}, Conf: {float(pt_conf[0]):.3f}\n")
        f.write(f"Triton  BBox: {triton_bbox_rounded}, Conf: {triton_conf:.3f}\n\n")

        f.write(f"Ground Truth Keypoints:\n{gt_kpts_rounded}\n")
        f.write(f"Pytorch Keypoints:\n{pt_kpts_rounded}\n")
        f.write(f"Triton  Keypoints:\n{triton_kpts_rounded}\n")

    print(f"Done. Results are saved at : {txt_path}")
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
