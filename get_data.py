import os
import zipfile
import tempfile

from pathlib import Path

import requests

def download_data(url:str,image_path:str):
  # Download pizza, steak, sushi data
  with tempfile.TemporaryFile() as fp:
    print("Downloading pizza, steak, sushi data...")
    r = requests.get(url,stream=True)
    fp.write(r.content)
    # Unzip pizza, steak, sushi data
    with zipfile.ZipFile(fp, "r") as zip_ref:
        print("Unzipping pizza, steak, sushi data...")
        zip_ref.extractall(image_path)
        print("Data unzipped")

url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
folder_name = "pizza_steak_sushi"

# Setup path to data folder
data_path = Path("data/")
image_path = data_path / folder_name

# If the image folder doesn't exist, download it and prepare it...
if image_path.is_dir():
    print(f"{image_path} directory exists.")
    if not os.listdir(image_path): # empty directory, download
      download_data(url=url,image_path=image_path)
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)
    download_data(url=url,image_path=image_path)
