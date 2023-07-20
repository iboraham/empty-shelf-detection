import os

from roboflow import Roboflow

def download_dataset(download_format:str="coco")->None:
    """
    Download dataset from Roboflow.

    Reference: https://docs.roboflow.com/api/download-dataset

    Warning:
        This function requires the environment variable *ROBOFLOW_API_KEY* to be set.
    """
    os.makedirs('data', exist_ok=True)

    rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))
    project = rf.workspace("fyp-ormnr").project("supermarket-empty-shelf-detector")
    dataset = project.version(3).download(download_format, location=os.path.join('data', 'Supermarket-Empty-Shelf-Detector--3'))

    # Rename dataset
    os.rename(dataset.location, dataset.location + download_format)
    

    print("Downloaded dataset to: ", dataset.location)


if __name__ == "__main__":
    download_dataset()