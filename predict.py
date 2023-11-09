# Predict - predict.py
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import v2 as transforms
import argparse
import model_builder
from PIL import Image

def predict(model:torch.nn.Module,num_images:int,image_path:str,transform:torchvision.transforms.v2=None,device=None):

  if not device:
    device = "cuda" if torch.cuda.is_available() else "cpu"
  
  model = model.to(device)

  # Get a random list of 3 image paths from test dataset
  import random
  num_imgs_to_plot = 3
  test_image_path_list = list(Path(test_dir).glob("*/*.jpg"))

  if not transform:
    # Transformer
    transform = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])

  for image_path in random.sample(test_image_path_list,k=num_imgs_to_plot):
    image = Image.open(image_path)
    image_tensor = transform(image).to(device)

    model.eval()
    with torch.inference_mode():
      y_logits = model(image_tensor.unsqueeze(dim=0))
      y_pred = torch.softmax(y_logits,dim=1).argmax(dim=1)
      confidence = torch.softmax(y_logits,dim=1).max(dim=1).values.item()
    plt.figure()
    plt.imshow(image)
    plt.axis(False)
    plt.title(f"{confidence*100:.2f}% | {class_names[y_pred]}")



if __name__ = '__main__':
  # TODO: class names as arg...
  class_names = ["pizza","steak","sushi"]

  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path",help="path of the model",type=str)
  parser.add_argument("--image_path",help="path of the image",type=str)

  # Load model (we have saved the entire model, not just the state dict!)

  args = parser.parse_args()

  # The utils.py --> save_model() saves the state_dict()
  # LOAD MODEL
  model = model_builder.TinyVGG(input_shape=3,output_shape=3,hidden_units=10)
  model.load_state_dict(torch.load(args.model_path))
  model.load_state_dict(torch.load(args.model_path))


  image_path = args.image_path

  # Transformer
  transform = transforms.Compose([
      transforms.Resize(size=(64,64)),
      transforms.ToTensor()
  ])

  image = Image.open(image_path)

  image_tensor = transform(image)
  print(image_tensor.shape)

  model.eval()
  with torch.inference_mode():
    y_logits = model(image_tensor.unsqueeze(dim=0))
    y_pred = torch.softmax(y_logits,dim=1).argmax(dim=1)
    confidence = torch.softmax(y_logits,dim=1).max(dim=1).values.item()

  print(f"I am {confidence*100:.2f}% sure that it's a {class_names[y_pred]}")
