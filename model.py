import torch
from torch import nn
from torch.nn import functional as F
import math
import os
import shutil
from sklearn.model_selection import train_test_split

#Input img -> Encoder (i.e go through hidden dimension) -> mean. std div -> parameterization (add noise) -> decoder -> output img
class VaritionalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()

        #encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)     #784 -> 200
        self.hid_2mu = nn.Linear(h_dim, z_dim)          #200 -> 20
        self.hid_2sigma = nn.Linear(h_dim, z_dim)       #200 -> 20

        #decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()


    def encode(self, x):
        #q_phi(z|x)
        h = self.relu(self.img_2hid(x))
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)
        return mu,sigma

    def decode(self, z):
        #p_theta(x|z)
        p = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(p))      #so value is betwn 0 & 1

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)       #reparamatrization
        z_new = mu + sigma*epsilon
        x_reconstructed = self.decode(z_new)
        return x_reconstructed, mu, sigma


if __name__ == "__main__":
    x = torch.randn(1, 28*28)
    vae = VaritionalAutoEncoder(input_dim=784)
    x_reconstructed, mu, sigma = vae(x)
    print(x_reconstructed.shape)
    print(mu.shape)
    print(sigma.shape)



