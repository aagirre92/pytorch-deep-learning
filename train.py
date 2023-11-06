"""
Trains a PyTorch image classification model using device-agnostic code
argparse module is used for passing hyperparameters via terminal
"""
import os
import argparse
from time import perf_counter
import torch
from torchvision.transforms import v2 as transforms
import data_setup, engine, model_builder,utils

parser = argparse.ArgumentParser()
parser.add_argument("--epochs",help="nº of epochs for training loop",type=int)
parser.add_argument("--batch_size",help="number of images per batch",type=int)
parser.add_argument("--hidden_units",help="hidden units for the model",type=int)
parser.add_argument("--learning_rate",help="Learning Rate for optimizer",type=float)
args = parser.parse_args()

# Setup hyperparameters
NUM_EPOCHS = args.epochs if args.epochs is not None else 5
BATCH_SIZE = args.batch_size if args.batch_size is not None else 32
HIDDEN_UNITS = args.hidden_units if args.hidden_units is not None else 10
LEARNING_RATE = args.learning_rate if args.learning_rate is not None else 0.001

# Setup directories
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms (you could create transforms.py as well)

data_transform = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.ToTensor()
])

# Create DataLoaders and get class names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_directory=train_dir,
                                                                               test_directory=test_dir,
                                                                               train_transform=data_transform,
                                                                               test_transform=data_transform,
                                                                               batch_size=BATCH_SIZE,
                                                                               )

# Create the model
model = model_builder.TinyVGG(input_shape=3,
                hidden_units=10,
                output_shape=len(class_names)).to(device)

# Set up loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(),lr=LEARNING_RATE)

# Start training the model
start_time = perf_counter()
results = engine.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=NUM_EPOCHS,
                       device=device)
end_time = perf_counter()
# Measure Training time
print(f"[INFO] Total training time: {end_time-start_time:.2f} seconds")
print(results)
# Save the model to file
utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_going_modular_script_mode_tinyvgg_model.pth")
