from ultralytics import YOLO
import os
import shutil
import onnx


def export_to_onnx(pt_path, imgsz=640, simplify=True, nms=True, dest_path=None):
    """
    Converts a trained YOLO model (.pt) into ONNX format for deployment.
    """

    # -----------------------------------------------------
    # Loading the trained YOLO model from .pt weights
    # -----------------------------------------------------
    print(f"Loading model from: {pt_path}")
    model = YOLO(pt_path)

    # -----------------------------------------------------
    # Exporting model to ONNX format with given parameters :
    # -----------------------------------------------------
    onnx_path = model.export(
        format="onnx",     
        imgsz=imgsz,       
        simplify=simplify,  # Simplifying ONNX graph to remove redundant operations
        nms=nms,            # Including Non-Max Suppression (post-processing) in ONNX graph
        opset=21,           # Operator set version for ONNX, we set newer opset (21) 
                            # which ensures compatibility
        dynamic=True       
    )

    print("Exported ONNX model:", onnx_path)

    # -----------------------------------------------------
    # Copying ONNX file to deployment repository
    # -----------------------------------------------------
    if dest_path:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(onnx_path, dest_path)
        print("Copied to:", dest_path)

    return onnx_path


if __name__ == "__main__":
    # -----------------------------------------------------
    # Defining paths for the model and ONNX destination
    # -----------------------------------------------------
    best_weight_path = "/mnt/g/My Drive/YOLO_dataset/results/train4/weights/best.pt"  # trained YOLOv11 weights
    dest_path = "/home/fidan/model_repository/yolo11/1/model.onnx"  # target ONNX save path

    # -----------------------------------------------------
    # Running export: .pt  to .onnx
    # -----------------------------------------------------
    export_to_onnx(
        best_weight_path,
        imgsz=640,       
        simplify=True, 
        nms=True,        
        dest_path=dest_path
    )
