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
    args.mode = 'test'
    
    # setting parameters
    FILE_NAME = f'{args.stock}_{args.name}'
    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    target_length = args.target_length // args.time_step
    
    test_datasets = StockDataset(args, f'./data/{args.stock}/test.csv')
    model_g = generator(test_datasets.num_features, args.noise_dim, target_length, device, args)
    
    # load best model and arguments
    check_point = torch.load(f'./model/{FILE_NAME}_best.pth')
    model_g.load_state_dict(check_point['model_g'])
    for key, value in vars(check_point['args']).items(): setattr(args, key, value)
    
    test_loader = DataLoader(test_datasets, args.batch_size, shuffle=False)
    print('------------------------------------------------------------------------------------------------')
    print(f'Evaluating model: {args.name}')
    print(f'Stock: {args.stock}')
    print('------------------------------------------------------------------------------------------------')
    plot_util = plot_predicions(path=f'./img/{FILE_NAME}', args=args)
    
    # predict
    y_preds = []
    for _ in range(args.pred_times):
        y_pred = predict(model_g, X, device, args)
        y_pred = test_datasets.scaler_y.inverse_transform(y_pred) # inverse pred transformation
        y_pred_arrange = arrange_time(y_pred, target_length, args.window_stride)[9:-9] # arrange list
        y_preds.append(y_pred_arrange)
    y_preds_arrange = [sum(x, []) for x in zip(*y_preds)] # 每天各個時間點股價預測值的陣列
    
    y_trues = test_datasets.scaler_y.inverse_transform(y) # inverse real value transformation
    y_trues_arrange = arrange_time(y_trues, target_length, args.window_stride)[9:-9] # arrange list
    # band plot
    plot_util.band_plot(np.array([y[0] for y in y_trues_arrange]), np.array(y_preds_arrange).flatten(), date)
    # dist plot
    chunk_size = len(y_preds_arrange) // args.num_days
    y_preds_arrange_split = [y_preds_arrange[i * chunk_size: (i + 1) * chunk_size] for i in range(args.num_days)]
    y_trues_arrange_split = [y_trues_arrange[i * chunk_size: (i + 1) * chunk_size] for i in range(args.num_days)]
    y_trues_arrange_dist = np.array([sum(parts, []) for parts in zip(*y_trues_arrange_split)])
    y_preds_arrange_dist = np.array([sum(parts, []) for parts in zip(*y_preds_arrange_split)])
    plot_util.dist_plot(y_trues_arrange_dist, y_preds_arrange_dist, date)
    # single date plot
    if args.num_days==1:
        plot_util.single_time_plot(np.array([y[0] for y in y_trues_arrange]), np.array(y_preds_arrange), date)