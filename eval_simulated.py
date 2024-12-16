# import packages
import os
import torch
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
# import file
from model.mygan import wgan
from model.rcgan import RCGAN
from model.forgan import ForGAN
from lib.calc import calc_kld
from arguments import parse_args
from simulated import DCCGARCHSimulator
from lib.visulization import plot_predicions
from lib.data import DataProcessor, StockDataset

def reshape_prediction_data(df):
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract date and time
    df['day'] = df['date'].dt.date
    df['time'] = df['date'].dt.time
    
    # Get unique days and times
    unique_days = sorted(df['day'].unique())
    unique_times = sorted(df['time'].unique())
    
    # Initialize the result array
    result = np.zeros((len(unique_days), len(unique_times), 100))
    
    # Fill the array
    for day_idx, day in enumerate(unique_days):
        for time_idx, time in enumerate(unique_times):
            mask = (df['day'] == day) & (df['time'] == time)
            values = df.loc[mask, 'Close'].values
            result[day_idx, time_idx, :len(values)] = values
    
    return result

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
    
    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    with open(f'./data/{args.stock}/scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)
    
    processor = DataProcessor(args, trial=int(args.name[-1]))
    test_datasets = StockDataset(args, f'./data/{args.stock}/test.csv')
    time_interval = test_datasets.time_intervals[args.window_size:]
    if args.model == 'mygan': model = wgan(test_datasets, args)
    elif args.model == 'forgan': model = ForGAN(test_datasets, args)
    elif args.model == 'rcgan': model = RCGAN(test_datasets, args)
    
    X = torch.tensor(np.array(test_datasets.X), dtype=torch.float32)
    y = np.array(test_datasets.y)
    y[y == -10] = np.nan
    
    plot_util = plot_predicions(path=f'./img/simulated_dist/{args.model}/{FILE_NAME}', args=args, time_interval=time_interval)
    
    # get predictions
    y_preds, _ = model.predict(X, y)
    
    # get data
    args.stock = args.stock.replace('_simulated', '')
    preprocessor = DataProcessor(args)
    data = preprocessor.get_data()
    # get simulations
    simulator = DCCGARCHSimulator(args, data)
    y_trues = simulator.simulate_close(time_interval, args.pred_times)
    y_trues = reshape_prediction_data(y_trues)
    
    for i, time_point in enumerate(time_interval):
        all_metrics = []
        for step in range(0, 1):
            print(f'Step: {step + 1}....')
            y_pred = y_preds[(i*9):(i+1)*9, step, :]
            # plot predictions
            plot_util.dist_simulate_plot(y_trues[i], y_pred, f'{time_point}_step_{step}')
            # calculate statistics
            metrics_df = calculate_metrics(y_trues[i], y_pred, scaler_y)
            metrics_df['prediction_step'] = step
            all_metrics.append(metrics_df)
    
        final_metrics = pd.concat(all_metrics, ignore_index=True)
        
        res_path = f'./res/{args.model}/{args.stock}_simulated'
        if not os.path.exists(res_path): os.makedirs(res_path)
        final_metrics.to_csv(f'{res_path}/{time_point}_metrics.csv', index=False)