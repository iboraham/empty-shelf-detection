from tidecv import TIDE, Data
from ultralytics import YOLO
import os
from tqdm import tqdm


def yolo2tide(root: str) -> list:
    """Convert a YOLO dataset to a TIDE ground truth array."""
    gt_data = []
    annotation_folder = os.path.join(root, "test", "labels")
    for filename in os.listdir(annotation_folder):
        with open(os.path.join(annotation_folder, filename), "r") as file:
            image_id = os.path.splitext(filename)[0]
            for line in file:
                class_id, x_center, y_center, width, height = map(float, line.split())
                bbox = [
                    x_center - width / 2,
                    y_center - height / 2,
                    width,
                    height,
                ]  # Convert to [x_min, y_min, width, height]
                gt_data.append(
                    {
                        "image_id": image_id,
                        "category_id": int(class_id),
                        "bbox": bbox,
                    }
                )
    return gt_data


def predict_single(model: YOLO, image: str) -> list:
    """Predict on an image and return a TIDE prediction array."""
    preds = model(image, verbose=False)[0]
    pred_data = []
    tbar = tqdm(
        zip(preds.boxes.xywh, preds.boxes.conf, preds.boxes.cls),
        total=len(preds.boxes),
        leave=False,
    )
    for (
        (x, y, w, h),
        (conf),
        (class_id),
    ) in (
        tbar
    ):  # Assuming preds.xywh contains the predictions in [x_center, y_center, width, height] format
        bbox = [x.item(), y.item(), w.item(), h.item()]
        bbox = [int(x) for x in bbox]
        pred_data.append(
            {
                "image_id": os.path.basename(image),
                "category_id": class_id.item(),
                "bbox": bbox,
                "score": conf.item(),
            }
        )
    return pred_data


predict_batch = lambda model, images: [
    predict_single(model, image)
    for image in tqdm(images, leave=False, desc="Predicting")
]

root = "../../datasets/Supermarket-Empty-Shelf-Detector--3yolov8"

tide = TIDE()
model = YOLO("empty-shelf-detection-yolov8/yolov8m/weights/best.pt")


# Read in the test set
gts = yolo2tide(root)
image_loc = os.path.join(root, "test", "images")
images = [os.path.join(image_loc, image) for image in os.listdir(image_loc)]
preds = predict_batch(model, images)


tide.evaluate(gts, preds, mode=TIDE.BOX)  # Use TIDE.MASK for masks
tide.summarize()  # Summarize the results as tables in the console
tide.plot(
    "."
)  # Show a summary figure. Specify a folder and it'll output a png to that folder.
