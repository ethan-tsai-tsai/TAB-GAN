import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math

from model.wgan_gp import WGAN_GP

if __name__ == '__main__':
    from preprocessor import DataProcessor
    from arguments import parse_args
    args = parse_args()
    data_path = f'./data/{args.stock}'
    data = pd.read_csv(f'{data_path}/train.csv')
    
    model = WGAN_GP()
    train_x = data.drop(['ts', 'y'], axis=1).values

    model.train_vae(train_x)