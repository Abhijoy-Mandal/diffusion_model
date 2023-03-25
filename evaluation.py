from re import T
import torch
from models import Unet
from dataset import ImagenetDataset
import numpy as np
import os
from matplotlib import pyplot as plt


def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    T = 1000

    model = Unet(T).to(device)
    state_dict = torch.load("model.pt")
    model.load_state_dict(state_dict)
    model.eval()

    dataset = ImagenetDataset(os.path.join("data", "train_data_batch_2"), T)

    data, e = dataset[1]
    x, y_tilde, e = data

    y = np.random.normal(0, 1, size=y_tilde.shape).astype("float32")

    images = []

    plt.imshow(x.transpose(1, 2, 0), cmap='gray')
    plt.show()

    

    x = torch.tensor(x, device=device, requires_grad=False)
    y = torch.tensor(y, device=device, requires_grad=False)

    x = x.unsqueeze(0)
    y = y.unsqueeze(0)


    for t in range(T-1, -1, -1):
        t_list = torch.tensor([t]).long().to(device)
        eps = model.forward(x, y, t_list)

        if t > 1:
            z = np.random.normal(0, 1, size=y.shape).astype("float32")
        else:
            z = np.zeros(y.shape, dtype=np.float32)
        z = torch.tensor(z, device=device)
        
        alpha, gamma = dataset.get_alpha(t)
        alpha = torch.tensor(alpha, device=device)
        gamma = torch.tensor(gamma, device=device)

        y = 1 / torch.sqrt(alpha) * (y - ((1 - alpha) / torch.sqrt(1 - gamma)) * eps)  + torch.sqrt(1 - alpha) * z

        if t % 100 == 0:
            images.append(y)
            
            plt.imshow(y.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
            plt.show()
        
        
    images.append(y)

    plt.imshow(y.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
    plt.show()



if __name__ == "__main__":
    main()