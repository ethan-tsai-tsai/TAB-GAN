# Import Package
import os
import torch
import numpy as np
# Import file
from model.mygan import wgan
from model.rcgan import RCGAN
from model.forgan import ForGAN
from arguments import parse_args
from preprocessor import StockDataset
from lib.visulization import plot_predicions

if __name__ == '__main__':
    # set arguments
    args = parse_args()
    FILE_NAME = f'{args.stock}_{args.name}'
    check_point = torch.load(f'./model_saved/{args.model}/{FILE_NAME}/final.pth')
    args.mode = 'test' # switch to test mode
    
    for key, value in check_point['args'].items(): 
        if key not in ['pred_times', 'bound_percent']:
            setattr(args, key, value)
    
    # setting parameters
    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    
    test_datasets = StockDataset(args, f'./data/{args.stock}/test.csv')
    time_interval = test_datasets.time_intervals[args.window_size:]
    
    if args.model == 'mygan': model = wgan(test_datasets, args)
    elif args.model == 'rcgan': model = RCGAN(test_datasets, args)
    elif args.model == 'forgan': model = ForGAN(test_datasets, args)
    
    # load best model and arguments
    model.model_g.load_state_dict(check_point['model_g'])
    
    X = torch.tensor(np.array(test_datasets.X), dtype=torch.float32)
    y = np.array(test_datasets.y)
    y[y == -10] = np.nan
    plot_util = plot_predicions(path=f'./img/{args.model}/{FILE_NAME}', args=args, time_interval=time_interval)
    print('------------------------------------------------------------------------------------------------')
    print(f'Evaluating model: {args.name}')
    print(f'Stock: {args.stock}')
    print('------------------------------------------------------------------------------------------------')
    
    # predict
    y_preds, y_trues = model.predict(X, y)
    plot_util.band_plot(y_trues, y_preds) # band plot
    plot_util.fixed_band_plot(y_trues, y_preds) # fixed band plot
    plot_util.dist_plot(y_trues, y_preds) # dist plot
    plot_util.fixed_dist_plot(y_trues, y_preds) # dist plot
    plot_util.single_time_plot(y_trues, y_preds) # single date plot
    plot_util.fixed_single_time_plot(y_trues, y_preds) # fixed single date plot