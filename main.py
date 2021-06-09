import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adagrad, Adam, SGD

from sklearn.model_selection import train_test_split
from models import *
from train import  train
from helpers import load_data


mlp = MLP(in_dim=8, out_dim=2, nb_hidden=4, hidden_dim=30)
optimizer = SGD(mlp.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
load_data('fashionMnist', 42)

