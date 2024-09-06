# Import Package
import os
import torch
import random
# Import file
from preprocessing import *
from model import *
from utils import *
from arguments import *
import pykalman

def predict(args, model_g, device, X, y=None):
    model_g.to(device)
    model_g.eval()
    with torch.inference_mode():
        X = X.unsqueeze(0).to(device)
        noise = torch.randn(X.shape[0], args.noise_dim).to(device)
        y_pred = model_g(X, noise).cpu().detach().tolist() # 輸出為三維
        y_pred = np.array(y_pred).flatten()
        if y is not None: y_true = y.cpu().detach().numpy()
        else: y_true = None
        return y_pred, y_true

def prepare_eval_data(model_g, stock_data, device, date, args, pred_times=1):
    eval_date = stock_data.time_intervals[stock_data.time_intervals.index(date):stock_data.time_intervals.index(date)+5]
    X, y = stock_data.get_data(date, days=args.num_days)
    # add noise
    X, y = X.to(device), y.to(device)
    y = y.unsqueeze(2)
    y_preds = []
    y_trues = []
    for i in range(X.shape[0]):
        for _ in range(pred_times):
            y_pred, y_true = predict(args, model_g, device, X[i], y[i])
            y_true = stock_data.scaler_y.inverse_transform(y_true)
            y_pred = stock_data.scaler_y.inverse_transform([y_pred])[0]
            y_preds.append(y_pred)
            y_trues.append(y_true)
        
    return eval_date, y_preds, y_trues

if __name__ == '__main__':
    args = parse_args()
    
    FILE_NAME = f'{args.stock}_{args.name}'
    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    stock_data = StockDataset(args, mode='test')
    model_g = generator(stock_data.num_features, args.noise_dim, stock_data.target_length, device)
    model_cp = torch.load(f'./model/{FILE_NAME}.pth')
    model_g.load_state_dict(model_cp['model_g'])
    
    # Predict last n
    print('------------------------------------------------------------------------------------------------')
    print(f'Evaluating model: {args.name}')
    print(f'Stock: {args.stock}')
    print('------------------------------------------------------------------------------------------------')
    # 清空資料夾內容
    if not os.path.exists(f'./img/pred/{FILE_NAME}'): os.makedirs(f'./img/pred/{FILE_NAME}')
    if not os.path.exists(f'./img/dist/{FILE_NAME}'): os.makedirs(f'./img/dist/{FILE_NAME}')
    clear_folder(f'./img/pred/{FILE_NAME}')
    clear_folder(f'./img/dist/{FILE_NAME}')
    eval_dates = random.sample(stock_data.time_intervals[args.num_days : -args.num_days], args.num_eval)
    for date in eval_dates:
        eval_date, y_preds, y_trues = prepare_eval_data(model_g, stock_data, device, date, args)
        save_predict_plot(args, f'./img/pred/{FILE_NAME}', f'pred_{eval_date[-1]}', eval_date, y_preds, y_trues)
        dist_eval_date, dist_y_preds, dist_y_trues = prepare_eval_data(model_g, stock_data, device, date, args, pred_times=100)
        save_dist_plot(args, f'./img/dist/{FILE_NAME}', f'dist_d{dist_eval_date[-1]}', dist_eval_date, dist_y_preds, dist_y_trues)