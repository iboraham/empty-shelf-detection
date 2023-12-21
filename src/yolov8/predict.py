import argparse
import os

from PIL import Image
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on a dataset")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(
            "empty-shelf-detection-yolov8", "yolov8m", "weights", "best.pt"
        ),
        help="Model to use.",
    )
    parser.add_argument(
        "-i",
        "--image_url",
        type=str,
        required=True,
        help="Image to predict on.",
    )
    return parser.parse_args()


def inference(args):
    # Load a model
    model = YOLO(args.checkpoint)

    image_url = args.image_url

    # Use the model
    results = model(
        image_url,
    )

    # Process results list
    for result in results:
        print(result)
        boxes = result.boxes  # Boxes object for bbox outputs
        probs = result.probs  # Probs object for classification outputs

        # Plot images with boxes and labels
        im_array = result.plot()
        im = Image.fromarray(im_array[..., ::-1])
        im.show()


if __name__ == "__main__":
    args = parse_args()
    inference(args)
