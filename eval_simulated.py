# import packages
import os
import torch
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
# import file
from model.mygan import wgan
from lib.calc import calc_kld
from arguments import parse_args
from simulated import DCCGARCHSimulator
from lib.visulization import plot_predicions
from preprocessor import DataProcessor, StockDataset

def calculate_metrics(y_true, y_pred, scaler_y):
    """Calculate RMSE, MAE, and KLD for each time step"""
    metrics = []
    
    for i in range(len(y_true)):
        true_dist = y_true[i]
        pred_dist = y_pred[i]
        
        rmse = np.sqrt(mean_squared_error(true_dist, pred_dist))
        mae = mean_absolute_error(true_dist, pred_dist)
        
        scale_true_dist = scaler_y.transform(true_dist.reshape(-1, 1))
        scale_pred_dist = scaler_y.transform(pred_dist.reshape(-1, 1))
        scale_true_dist = scale_true_dist.ravel()
        scale_pred_dist = scale_pred_dist.ravel()
        
        kld = calc_kld(scale_pred_dist, scale_true_dist)
        
        metrics.append({
            'time_step': i,
            'RMSE': rmse,
            'MAE': mae,
            'KLD': kld
        })
    
    return pd.DataFrame(metrics)

if __name__ == '__main__':
    # set arguments
    args = parse_args()
    FILE_NAME = f'{args.stock}_{args.name}'
    check_point = torch.load(f'./model_saved/{args.model}/{FILE_NAME}/final.pth')
    args.mode = 'test'
    
    for key, value in check_point['args'].items(): 
        if key not in ['pred_times', 'bound_percent']:
            setattr(args, key, value)
    
    # setting parameters
    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    with open(f'./data/{args.stock}/scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)
    
    processor = DataProcessor(args, trial=1)
    test_datasets = StockDataset(args, f'./data/{args.stock}/test.csv')
    time_interval = test_datasets.time_intervals[args.window_size:]
    wgan_model = wgan(test_datasets, args)
    
    X = torch.tensor(np.array(test_datasets.X), dtype=torch.float32)
    y = np.array(test_datasets.y)
    y[y == -10] = np.nan
    
    plot_util = plot_predicions(path=f'./img/simulated_dist/{FILE_NAME}', args=args, time_interval=time_interval)
    
    # get predictions
    X = X[:9]
    y_preds, _ = wgan_model.predict(X, y)
    # get data
    args.stock = args.stock.replace('_simulated', '')
    preprocessor = DataProcessor(args)
    data = preprocessor.get_data()
    # get simulations
    simulator = DCCGARCHSimulator(args, data)
    y_trues = simulator.simulate_close(time_interval[0], 1000)
    y_trues = y_trues.groupby('date')['Close'].apply(lambda x: x.values).values
    y_trues = np.array([i.flatten() for i in y_trues])
    
    all_metrics = []
    for step in range(0, 1):
        print(f'Step: {step + 1}....')
        y_pred = y_preds[:, step, :]
        # plot predictions
        plot_util.dist_simulate_plot(y_trues, y_pred, f'{time_interval[0]}_step_{step}')
        # calculate statistics
        metrics_df = calculate_metrics(y_trues, y_pred, scaler_y)
        metrics_df['prediction_step'] = step
        all_metrics.append(metrics_df)
    
    final_metrics = pd.concat(all_metrics, ignore_index=True)
    
    res_path = f'./res/{args.stock}_simulated'
    if not os.path.exists(res_path): os.makedirs(res_path)
    final_metrics.to_csv(f'{res_path}/{time_interval[0]}_metrics.csv', index=False)