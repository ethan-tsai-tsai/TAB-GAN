import os
import torch
import pickle
import numpy as np
from torch import nn
from datetime import datetime
from torch.utils.data import DataLoader, random_split
# Import file
from model.mygan import wgan
from arguments import parse_args
from lib.utils import save_model
from preprocessor import StockDataset
from lib.visulization import plot_predicions, save_loss_curve

if __name__ == '__main__':
    
    start_time = datetime.now()
    
    # set arguments
    args = parse_args()
    args.mode = 'train'
    file_name = f'./model_saved/{args.stock}_{args.name}/bayes_args.pkl'
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            saved_args = pickle.load(f)
        for key, value in saved_args.items():
            if hasattr(args, key):
                setattr(args, key, value)
    train_datasets = StockDataset(args, f'./data/{args.stock}/train.csv')
    train_size = int(0.95 * len(train_datasets))
    val_size = len(train_datasets) - train_size
    train_data, val_data = random_split(train_datasets, [train_size, val_size])
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    val_datasets = StockDataset(args, f'./data/{args.stock}/test.csv')
    wgan_model = wgan(train_datasets, args)
    plot_util = plot_predicions(f'./logs/{args.stock}_{args.name}', args, val_datasets.time_intervals)
    
    print('----------------------------------------------------------------')
    print('Start training...')
    print('----------------------------------------------------------------')
    print('hyperparameters: ')
    filter_val = ['noise_dim', 
                  'epoch', 'batch_size', 
                  'hidden_dim_g', 'num_layers_g', 'num_head_g', 'lr_g',
                  'hidden_dim_d', 'num_layers_d', 'num_head_d', 'lr_d',
                  'd_iter', 'gp_lambda']
    for k, v in vars(args).items():
        if k in filter_val: print("{}:\t{}".format(k, v))
    print('----------------------------------------------------------------')
    results = wgan_model.train(train_loader, val_loader)
    save_loss_curve(results, args)
    file_name = f'./model/{args.stock}_{args.name}/final.pth'
    save_model(wgan_model.model_d, wgan_model.model_g, args, file_name)
    
    end_time = datetime.now()
    training_time = (end_time - start_time).seconds
    print(f'Training time: {training_time//3600}:{(training_time%3600)//60}:{training_time%60}')