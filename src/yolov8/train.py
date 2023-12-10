import wandb
from ultralytics import YOLO
import os


def train():
    # Initialize wandb
    run = wandb.init(
        project="empty-shelf-detection",
        config={
            "epochs": 15,
            "model": "yolov8m",
            "dataset": "../../datasets/Supermarket-Empty-Shelf-Detector--3yolov8/data.yaml",
            "device": 0,
            "image_size": 640,
            "freeze_layers": 5,
        },
    )

    # Load a model
    model = YOLO(run.config.model + ".pt")

    # Use the model
    model.train(
        data=run.config.dataset,
        epochs=run.config.epochs,
        device=run.config.device,
        seed=42,
        project="empty-shelf-detection-yolov8",
        name=run.config.model,
        imgsz=run.config.image_size,
        exist_ok=True,
        freeze=run.config.freeze_layers,
    )

    metrics = model.val()  # evaluate model performance on the validation set

    # Log metrics to wandb
    wandb.log(metrics)

    run.finish()  # This is needed to tell wandb that the run has ended


if __name__ == "__main__":
    wandb.login(key=os.environ["WANDB_API_KEY"])
    train()
