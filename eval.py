# Import Package
import os
import torch
import random
# Import file
from preprocessing import *
from model import *
from utils import *
from arguments import *

def predict(model_g, X, device, args):
    # predict
    model_g.to(device)
    model_g.eval()
    with torch.inference_mode():
        noise = torch.randn(X.shape[0], args.noise_dim).to(device)
        y_preds = model_g(X, noise).squeeze().cpu().detach().numpy()
    return y_preds

if __name__ == '__main__':
    args = parse_args()
    
    FILE_NAME = f'{args.stock}_{args.name}'
    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    stock_data = StockDataset(args, mode='test')
    model_g = generator(stock_data.num_features, args.noise_dim, stock_data.target_length, device, args)
    model_cp = torch.load(f'./model/{FILE_NAME}_best.pth')
    # model_cp = torch.load(f'./model/{FILE_NAME}.pth')
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
        # prepare eval data
        time_idx = stock_data.time_intervals.index(date)
        eval_date = stock_data.time_intervals[time_idx:time_idx+args.num_days] # the dates of evaluation in one plot
        X, y = stock_data.get_data(date, days=args.num_days) 
        X = X.to(device)
        # predict
        y_preds = predict(model_g, X, device, args)
        # inverse transformation
        y_preds = stock_data.scaler_y.inverse_transform(y_preds)
        y_trues = stock_data.scaler_y.inverse_transform(y)
        print(y_preds.shape)
        y_preds_arrange = arrange_time(y_preds, args.target_length//args.time_step, args.window_stride)
        print(len(y_preds_arrange))
        # save_predict_plot(args, f'./img/pred/{FILE_NAME}', f'pred_{eval_date[-1]}', eval_date, y_preds, y_trues)
        # save_dist_plot(args, f'./img/dist/{FILE_NAME}', f'dist_{eval_date[0]}_to_{eval_date[-1]}', eval_date, y_preds, y_trues)