import torch
from torchvision import datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pickle
import numpy as np
from timeit import default_timer as timer
import cv2

import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

class ImagenetDataset(Dataset):
    def __init__(self, file_name, T):
        x, mean = load_data(file_name)
        print("data read from file")
        self.T = T
        #data processing
        #t, y = None, None
        self.alphas, self.alpha_bars = Alpha(self.T)
        self.x_train = x
        self.data_len = self.x_train.shape[0]

    def __str__(self):
        return f'Data Length (y_i size): {self.data_len} \nNumber of steps T: {self.T} \n{self.alpha_bars}'

    def __len__(self):
        return 40 * 32
        # return len(self.x_train)

    def __getitem__(self, idx):
        t = np.random.randint(0, self.T)
        i = idx%self.data_len

        x = self.grayscale(self.x_train[i])
        y, e = self.noise(self.x_train[i], t)
        return (x, y, t), e

    def grayscale(self, image):
        image = np.array(cv2.cvtColor(image.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY))
        image = np.expand_dims(image, axis=0)
        return image

    def get_alpha(self, t):
        return self.alphas[t], self.alpha_bars[t]

    def noise(self, image, t):
        '''
        takes an image, the timestamp t to add noise for out of T steps.
        '''
        if t==-1:
            return torch.tensor(image, dtype=torch.float32)
        alpha=self.alpha_bars[t]
        e = np.random.normal(loc=0, scale=(1-alpha), size=image.shape)
        e = e.astype("float32")
        image = np.sqrt(alpha)*image
        return torch.tensor((e + image), dtype=torch.float32), e

def preprocessing(x, T=1000, root="./Data/Batch1"):
    '''
    Assuming root exists, creates
    '''
    np.save(f"{root}/data_0.npy", x)
    for t in range(T):
        print(t*1.0/T)
        beta = Beta(t*1.0, T)
        y_t_1 = np.load(f"{root}/data_{t}.npy")
        y_t_2 =np.sqrt(beta)*np.random.normal(loc=0, scale=1, size=y_t_1.shape) + np.sqrt(1-beta)*y_t_1
        np.save(f"{root}/data_{t+1}.npy", y_t_2)

def Beta(t, T):
    '''
    calculates rate of noising for a given timestamp t out ot T
    '''
    return t/T*0.2+(T-t)/T*10e-4

def Alpha(T):
    betas = np.fromfunction(lambda t: (t + 1) / T * 0.2 - ((t + 1) - T) / T * 10e-4, shape=(T,))
    alphas = 1 - betas

    alpha_bars = np.zeros_like(alphas)
    alpha_bars[0] = alphas[0]
    for s in range(1, T):
        alpha_bars[s] = alpha_bars[s - 1] * alphas[s]
    return alphas, alpha_bars

def load_data(path, threads=5):
    '''
    unpickles and loads imagenet data from path
    '''
    data = unpickle(path)
    mean_image = data['mean']
    # x = data['data']
    data["data"] = data["data"] / np.float32(255)
    mean_image = mean_image / np.float32(255)
    #x -= mean_image
    img_size = 64
    img_size2 = img_size * img_size

    data["data"] = np.dstack((data["data"][:, :img_size2], data["data"][:, img_size2:2 * img_size2], data["data"][:, 2 * img_size2:]))
    data["data"] = data["data"].reshape((data["data"].shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)
    return data["data"], mean_image.reshape(img_size, img_size, 3)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def get_image(data, index, mean):
    '''
    gets image from data that is normalizesd my mean at index.
    '''
    print(data.shape)
    image = data[index]+mean
    plt.imshow(image)
    plt.show()

def get_image_no_mean(data, index):
    '''
    gets image for images that are not normalized
    '''
    get_image(data, index, 0)

def load_noised_data(root, T):
    '''
    Loads all steps of data from root and returns a list of size T+1 sets of images
    '''
    out = []
    for t in range(T+1):
        data = np.load(f"{root}/data_{t}.npy")
        data = data.transpose(0, 2, 3, 1)
        print(data.shape)
        out.append([data])
    return out

def check_preprocessed(root, T):
    '''
    Provides functionality to load all steps in a batch and view the image
    '''
    data = load_noised_data(root, T)
    while (1):
        stage = int(input("enter stage to view (-1 exit)"))
        index = int(input("enter index to view (-1 exit)"))
        if index == -1:
            break
        get_image_no_mean(data[stage][0], index)

def start_preprocessing(path, num_batches=10, T=10):
    '''
    Assuming ./Data exists, it creates ./Data/Batch{i} containing T steps of noising, each step as a
    .npy file.
    @params
    path: path to imagenet dataset
    num_batches: number of batches to divide the image data into
    T: number of noising steps
    '''
    data, mean = load_data(path)
    for i in range(num_batches):
        preprocessing(data[:data.shape[0] // num_batches], T=T, root=f"./Data/Batch{i+1}")

if __name__=="__main__":
    #start_preprocessing('C:\\Users\\Dell\\Downloads\\Imagenet64\\train_data_batch_2', num_batches=100, T=100)
    # check_processed("./Data", 10)
    print("loading dataset from file...")
    dataset= ImagenetDataset(os.path.join("data", "train_data_batch_1"), 1000)
    print("loaded data\nTesting time required for 128000")
    print(dataset)
    # alphas = Alpha(1000)
    # print(alphas)


