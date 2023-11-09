"""
Trains a PyTorch image classification model using device-agnostic code
argparse module is used for passing hyperparameters via terminal
"""
import os
import argparse
from time import perf_counter
import torch
from torchvision.transforms import v2 as transforms
import data_setup
import engine
import model_builder
import utils
from engine import train_step, test_step
from typing import Dict, List, Tuple
from tqdm.auto import tqdm


def train_with_writer(model: torch.nn.Module,
                      train_dataloader: torch.utils.data.DataLoader,
                      test_dataloader: torch.utils.data.DataLoader,
                      optimizer: torch.optim.Optimizer,
                      loss_fn: torch.nn.Module,
                      epochs: int,
                      writer: torch.utils.tensorboard.SummaryWriter,
                      device: torch.device) -> Dict[str, List[float]]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
      model: A PyTorch model to be trained and tested.
      train_dataloader: A DataLoader instance for the model to be trained on.
      test_dataloader: A DataLoader instance for the model to be tested on.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      loss_fn: A PyTorch loss function to calculate loss on both datasets.
      epochs: An integer indicating how many epochs to train for.
      writer: Custom writer for TensorBoard display.
      device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
      A dictionary of training and testing loss as well as training and
      testing accuracy metrics. Each metric has a value in a list for
      each epoch.
      In the form: {train_loss: [...],
                    train_acc: [...],
                    test_loss: [...],
                    test_acc: [...]}
      For example if training for epochs=2:
                   {train_loss: [2.0616, 1.0537],
                    train_acc: [0.3945, 0.3945],
                    test_loss: [1.2641, 1.5706],
                    test_acc: [0.3400, 0.2973]}
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)

        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        ### NEW: EXPERIMENT TRACKING FOR TENSORBOARD ###

        writer.add_scalars(main_tag="Loss",
                           tag_scalar_dict={"train_loss": train_loss,
                                            "test_loss": test_loss},
                           global_step=epoch)

        writer.add_scalars(main_tag="Accuracy",
                           tag_scalar_dict={"train_acc": train_acc,
                                            "test_acc": test_acc},
                           global_step=epoch)

        writer.add_graph(model=model,
                         input_to_model=torch.randn(32, 3, 224, 224).to(device))  # TRACK COMPUTATION GRAPH

        # Close the writer
        writer.close()

        ### END NEW ###

    # Return the filled results at the end of the epochs
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", help="nº of epochs for training loop", type=int)
    parser.add_argument(
        "--batch_size", help="number of images per batch", type=int)
    parser.add_argument(
        "--hidden_units", help="hidden units for the model", type=int)
    parser.add_argument("--learning_rate",
                        help="Learning Rate for optimizer", type=float)
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
        transforms.Resize(size=(64, 64)),
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
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

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
