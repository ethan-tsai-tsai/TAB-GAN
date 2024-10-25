# Import Package
import os
import torch
import argparse
from torch.utils.data import DataLoader
# Import file
from preprocessor import *
from model import *
from utils import *
from arguments import *
from train import wgan

if __name__ == '__main__':
    # set arguments
    args = parse_args()
    FILE_NAME = f'{args.stock}_{args.name}'
    check_point = torch.load(f'./model/{FILE_NAME}/final.pth')
    args.mode = 'test' # switch to test mode
    
    if os.path.exists(f'./model/{FILE_NAME}_args.pkl'):
        with open(f'./model/{FILE_NAME}_args.pkl', 'rb') as f:
            saved_args = pickle.load(f)
        for key, value in saved_args.items():
            if hasattr(args, key):
                setattr(args, key, value)
    else:
        for key, value in check_point['args'].items(): 
            if key not in ['pred_times', 'bound_percent']:
                setattr(args, key, value)
    
    # setting parameters
    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    target_length = args.target_length // args.time_step
    
    test_datasets = StockDataset(args, f'./data/{args.stock}/test.csv')
    time_interval = test_datasets.time_intervals[args.window_size + 1:]
    wgan_model = wgan(test_datasets, args)
    
    # load best model and arguments
    wgan_model.model_g.load_state_dict(check_point['model_g'])
    
    X = torch.tensor(np.array(test_datasets.X), dtype=torch.float32)
    y = np.array(test_datasets.y)
    plot_util = plot_predicions(path=f'./img/{FILE_NAME}', args=args, time_interval=time_interval)
    print('------------------------------------------------------------------------------------------------')
    print(f'Evaluating model: {args.name}')
    print(f'Stock: {args.stock}')
    print('------------------------------------------------------------------------------------------------')
    
    # predict
    y_preds, y_trues = wgan_model.predict(X, y)
    plot_util.band_plot(y_trues, y_preds) # band plot
    plot_util.fixed_band_plot(y_trues, y_preds) # fixed band plot
    plot_util.dist_plot(y_trues, y_preds) # dist plot
    plot_util.fixed_dist_plot(y_trues, y_preds) # dist plot
    plot_util.single_time_plot(y_trues, y_preds) # single date plot
    plot_util.fixed_single_time_plot(y_trues, y_preds) # fixed single date plot