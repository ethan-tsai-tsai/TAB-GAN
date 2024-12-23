import os
import pickle
from datetime import datetime
from torch.utils.data import DataLoader, random_split
# Import file
from model.rcgan import RCGAN
from model.tabgan import TABGAN
from model.forgan import ForGAN
from arguments import parse_args
from lib.data import StockDataset, SubsetStockDataset

if __name__ == '__main__':
    # record training time
    start_time = datetime.now()
    
    # set arguments
    args = parse_args()
    args.mode = 'train'
    model_path = f'./model_saved/{args.model}/{args.stock}_{args.name}'
    file_name = f'{model_path}/bayes_args.pkl'
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            saved_args = pickle.load(f)
        for key, value in saved_args.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # set dirs
    if not os.path.exists(f'./model_saved'): os.makedirs('./model_saved')
    if not os.path.exists(f'./model_saved/{args.model}'): os.makedirs(f'./model_saved/{args.model}')
    if not os.path.exists(model_path): os.makedirs(model_path)
    if not os.path.exists('./img'): os.makedirs('./img')
    if not os.path.exists(f'./img/{args.model}'): os.makedirs(f'./img/{args.model}')
    if not os.path.exists(f'./img/{args.model}/{args.stock}_{args.name}'): os.makedirs(f'./img/{args.model}/{args.stock}_{args.name}')
    # prepare dataset
    train_datasets = StockDataset(args, f'./data/{args.stock}/train.csv')
    
    train_size = int(0.95 * len(train_datasets))
    val_size = len(train_datasets) - train_size
    train_data, val_data = random_split(train_datasets, [train_size, val_size])
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    
    if args.model == 'tabgan':
        model = TABGAN(train_datasets, args)
    elif args.model == 'rcgan':
        model = RCGAN(train_datasets, args)
    elif args.model == 'forgan':
        model = ForGAN(train_datasets, args)
    
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
    if args.model in ['tabgan', 'rcgan']:
        results = model.train(train_loader, val_loader)
    elif args.model == 'forgan':
        train_indices, val_indices = random_split(range(len(train_datasets)), [train_size, val_size])
        train_data = SubsetStockDataset(train_datasets, train_indices.indices)
        val_data = SubsetStockDataset(train_datasets, val_indices.indices)
        model.train(train_data, val_data)
    
    end_time = datetime.now()
    training_time = (end_time - start_time).seconds
    print(f'Training time: {training_time//3600}:{(training_time%3600)//60}:{training_time%60}')