import wandb
from ultralytics import YOLO
import os


def train():
    # Initialize wandb
    run = wandb.init(
        project="empty-shelf-detection",
        config={
            "epochs": 500,
            "patience": 50,
            "model": "yolov8m",
            "dataset": "../../datasets/Supermarket-Empty-Shelf-Detector--3yolov8/data.yaml",
            "device": 0,
            "image_size": 960,
            "freeze_layers": None,
            "batch_size": -1,
            "learning_rate": 0.0001,
        },
    )

    # Load a model
    model = YOLO(run.config.model + ".pt")

    # Use the model
    model.train(
        data=run.config.dataset,
        epochs=run.config.epochs,
        patience=run.config.patience,
        device=run.config.device,
        seed=42,
        project="empty-shelf-detection-yolov8",
        name=run.config.model,
        imgsz=run.config.image_size,
        exist_ok=True,
        freeze=run.config.freeze_layers,
        batch=run.config.batch_size,
        lr0=run.config.learning_rate,
    )

    metrics = model.val(split="test")  # evaluate model performance on the test set

    # Log metrics
    wandb.log(metrics)

    run.finish()  # This is needed to tell wandb that the run has ended


if __name__ == "__main__":
    wandb.login(key=os.environ["WANDB_API_KEY"])
    train()
