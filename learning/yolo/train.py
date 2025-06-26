from ultralytics import YOLO

DATASET_CONFIG_PATH = "workspace/yolo/yolo.yaml"
MODEL_PATH= "workspace/checkpoints/yolo11x-seg.pt"
SAVE_DIR = "output/yolo"

def train_yolo():
    model = YOLO(MODEL_PATH)

    results = model.train(
        data=DATASET_CONFIG_PATH, 
        epochs=50,                
        batch=16,                 
        imgsz=600,                
        project=SAVE_DIR,         
        name="custom_training",   
        verbose=True              
    )

    print("Обучение завершено.")
    print(f"Сохраненные веса: {results['best']}")

if __name__ == "__main__":
    train_yolo()