import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from models import Unet
from dataset import ImagenetDataset
import os


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # Hyperparameters
    lr = 0.003
    batch_size = 32
    T = 1000

    # Model Initialization
    model = Unet(T).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.MSELoss()

    print("Model initialized")

    # Load Data
    train_dataset = ImagenetDataset(os.path.join("data", "train_data_batch_1"), T)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print("Data Loaded")

    # Training
    model.train()
    loss_list = []

    for data, e in iter(train_dataloader):
        x, y_tilde, t = data

        x = torch.tensor(x, device=device)
        y_tilde = torch.tensor(y_tilde, device=device)
        t = torch.tensor(t, device=device)
        e = torch.tensor(e, device=device)

        pred_eps = model(x, y_tilde, t)

        loss = criterion(pred_eps, e)

        # Grad Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        loss_list.append(loss.cpu())

        print(f"training loss: {loss}")
    
        # Save Model
        torch.save(model.state_dict(), "model.pt")
    
    # Plot loss
    plt.figure()
    plt.plot(loss_list)
    plt.savefig("training_loss.png")
    




if __name__ == "__main__":
    main()

