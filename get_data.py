import os
import zipfile
import tempfile

from pathlib import Path

import requests


def get_data(url: str, folder_name: str):

    def download_data(url: str, image_path: str):
        # Download data
        with tempfile.TemporaryFile() as fp:
            print("Downloading data...")
            r = requests.get(url, stream=True)
            fp.write(r.content)
            # Unzip data
            with zipfile.ZipFile(fp, "r") as zip_ref:
                print("Unzipping data...")
                zip_ref.extractall(image_path)
                print("Data unzipped")

    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / folder_name

    # If the image folder doesn't exist, download it and prepare it...
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
        if not os.listdir(image_path):  # empty directory, download
            download_data(url=url, image_path=image_path)
    else:
        print(f"Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        download_data(url=url, image_path=image_path)

    return image_path
