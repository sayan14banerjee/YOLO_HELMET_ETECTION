from ultralytics import YOLO
import os

def train_yolov8(
        data_yaml="data/data.yaml",   # ✅ correct path now
        model_name="yolov8n.pt",
        img_size=640,
        epochs=50,
        project="runs/detect",
        name="train",
        save_dir="models/"
    ):
    """
    Train YOLOv8 model with given configs
    """
    # Load pretrained YOLOv8 backbone
    model = YOLO(model_name)

    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        project=project,
        name=name
    )

    # Save best weights
    best_model_path = os.path.join(save_dir, "best.pt")
    os.makedirs(save_dir, exist_ok=True)
    model.save(best_model_path)

    print(f"✅ Training completed. Best model saved at: {best_model_path}")

if __name__ == "__main__":
    train_yolov8()