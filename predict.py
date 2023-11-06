# Predict - predict.py
import torch
from PIL import Image
from torchvision.transforms import v2 as transforms
import argparse
import model_builder

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
