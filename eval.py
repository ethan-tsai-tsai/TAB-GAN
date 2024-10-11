# Import Package
import os
import torch
import random
from torch.utils.data import DataLoader
# Import file
from preprocessor import *
from model import *
from utils import *
from arguments import *
from train import wgan

if __name__ == '__main__':
    args = parse_args()
    args.mode = 'test' # switch to test mode
    
    # setting parameters
    FILE_NAME = f'{args.stock}_{args.name}'
    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    target_length = args.target_length // args.time_step
    
    test_datasets = StockDataset(args, f'./data/{args.stock}/test.csv')
    model_g = generator(test_datasets.num_features - 1, args.noise_dim, target_length, device, args)
    
    # load best model and arguments
    check_point = torch.load(f'./model/{FILE_NAME}_best.pth')
    model_g.load_state_dict(check_point['model_g'])
    for key, value in vars(check_point['args']).items(): setattr(args, key, value)
    
    test_loader = DataLoader(test_datasets, args.batch_size, shuffle=False)
    plot_util = plot_predicions(path=f'./img/{FILE_NAME}', args=args, time_interval=test_datasets.time_intervals)
    print('------------------------------------------------------------------------------------------------')
    print(f'Evaluating model: {args.name}')
    print(f'Stock: {args.stock}')
    print('------------------------------------------------------------------------------------------------')
    
    # predict
    wgan_model = wgan(test_datasets, args)
    y_preds, y_trues = wgan_model.predict(test_loader)
    plot_util.band_plot(y_trues, y_preds) # bound plot
    plot_util.dist_plot(y_trues, y_preds) # dist plot
    plot_util.single_time_plot(y_trues, y_preds) # single date plot