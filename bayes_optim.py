import os
import optuna
import pickle
import logging
from datetime import datetime
from torch.utils.data import DataLoader, random_split

from model.rcgan import RCGAN
from model.tabgan import TABGAN
from model.forgan import ForGAN
from arguments import parse_args
from lib.data import StockDataset, SubsetStockDataset

optuna.logging.set_verbosity(optuna.logging.DEBUG)
def tabgan_objective(trial):
    try:
        # set hyperparameters
        args.mode = 'optim' # optim mode
        # data
        args.noise_dim = trial.suggest_categorical('noise_dim', [32, 64, 128, 256])
        # model
        args.hidden_dim_g = trial.suggest_categorical('hidden_dim_g', [32, 64, 128, 256])
        args.num_layers_g = trial.suggest_int('num_layers_g', 1, 4)
        args.num_head_g = trial.suggest_categorical('num_head_g', [4, 8, 16])
        args.hidden_dim_d = trial.suggest_categorical('hidden_dim_d', [16, 32, 64, 128, 256])
        args.num_layers_d = trial.suggest_int('num_layers_d', 1, 4)
        args.num_head_d = trial.suggest_categorical('num_head_d', [4, 8, 16])
        # train
        args.epoch  = trial.suggest_int('epoch', 100, 300)
        args.lr_d = trial.suggest_float('lr_d', 1e-6, 1e-2, log=True)
        args.lr_g = trial.suggest_float('lr_g', 1e-6, 1e-2, log=True)
        args.batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
        args.d_iter = trial.suggest_int('d_iter', 1, 5)
        args.gp_lambda = trial.suggest_int('gp_lambda', 1, 10)
        
        # prepare dataset
        train_datasets = StockDataset(args, f'./data/{args.stock}/train.csv')
        train_size = int(0.95 * len(train_datasets))
        val_size = len(train_datasets) - train_size
        train_data, val_data = random_split(train_datasets, [train_size, val_size])
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
        # val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
        test_datasets = StockDataset(args, f'./data/{args.stock}/test.csv')
        val_loader = DataLoader(test_datasets, batch_size=args.batch_size, shuffle=False)
        print(f'Starting trial with params: {trial.params}')
        # model setup
        model = TABGAN(train_datasets, args)
        _ = model.train(train_loader, val_loader)
        _, _, test_kld = model.validation(val_loader)
        test_score = test_kld
        score = test_score
        return score
    except Exception as e:
        logging.error(f"Error in trial {trial.number}: {e}")
        raise optuna.exceptions.TrialPruned()

def rcgan_objective(trial):
    try:
        # set hyperparameters
        args.mode = 'optim' # optim mode
        # data
        args.noise_dim = trial.suggest_categorical('noise_dim', [32, 64, 128, 256])
        # model
        args.hidden_dim_g = trial.suggest_categorical('hidden_dim_g', [32, 64, 128, 256])
        args.num_layers_g = trial.suggest_int('num_layers_g', 1, 4)
        args.hidden_dim_d = trial.suggest_categorical('hidden_dim_d', [16, 32, 64, 128, 256])
        args.num_layers_d = trial.suggest_int('num_layers_d', 1, 4)
        # train
        args.epoch  = trial.suggest_int('epoch', 100, 500)
        args.lr_d = trial.suggest_float('lr_d', 1e-6, 1e-2, log=True)
        args.lr_g = trial.suggest_float('lr_g', 1e-6, 1e-2, log=True)
        args.batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
        
        # prepare dataset
        train_datasets = StockDataset(args, f'./data/{args.stock}/train.csv')
        train_size = int(0.95 * len(train_datasets))
        val_size = len(train_datasets) - train_size
        train_data, val_data = random_split(train_datasets, [train_size, val_size])
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
        
        print(f'Starting trial with params: {trial.params}')
        # model setup
        model = RCGAN(train_datasets, args)
        _ = model.train(train_loader, val_loader)
        _, _, test_kld = model.validation(val_loader)
        test_score = test_kld
        score = test_score
        return score
    except Exception as e:
        logging.error(f"Error in trial {trial.number}: {e}")
        raise optuna.exceptions.TrialPruned()
        
def forgan_objective(trial):
    try:
        # set hyperparameters
        args.mode = 'optim' # optim mode
        # data
        args.cell_type = trial.suggest_categorical('cell_type', ['gru', 'lstm'])
        # model
        args.hidden_dim_g = trial.suggest_categorical('hidden_dim_g', [1, 2, 4, 8, 16, 32, 64, 128, 256])
        args.hidden_dim_d = trial.suggest_categorical('hidden_dim_d', [1, 2, 4, 8, 16, 32, 64, 128, 256])
        args.noise_dim = trial.suggest_categorical('noise_dim', [1, 2, 4, 8, 16, 32])
        args.d_iter = trial.suggest_int('d_iter', 1, 7)

        args.epoch = trial.suggest_int('epoch', 500, 500)
        args.batch_size = trial.suggest_int('batch_size', 1000, 1000)
        args.lr_g = trial.suggest_float('lr_g', 1e-3, 1e-3, log=True)
        args.lr_d = trial.suggest_float('lr_d', 1e-3, 1e-3, log=True)
        
        # prepare dataset
        train_datasets = StockDataset(args, f'./data/{args.stock}/train.csv')
        train_size = int(0.95 * len(train_datasets))
        val_size = len(train_datasets) - train_size
        train_indices, val_indices = random_split(range(len(train_datasets)), [train_size, val_size])
        train_data = SubsetStockDataset(train_datasets, train_indices.indices)
        val_data = SubsetStockDataset(train_datasets, val_indices.indices)
        val_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
        print(f'Starting trial with params: {trial.params}')
        # model setup
        forgan = ForGAN(train_datasets, args)
        forgan.train(train_data, val_data)
        _, _, test_kld = forgan.validation(val_loader)
        score = test_kld
        return score
    except Exception as e: 
        logging.error(f"Error in trial {trial.number}: {e}")
        return float('inf')

if __name__ == '__main__':
    
    start_time = datetime.now()
    
    args = parse_args()
    model_path = f'./model_saved/{args.model}/{args.stock}_{args.name}'
    if not os.path.exists(f'./model_saved/{args.model}'): os.makedirs(f'./model_saved/{args.model}')
    if not os.path.exists(model_path): os.makedirs(model_path)
    
    study = optuna.create_study(direction='minimize')
    if args.model == 'tabgan':
        study.optimize(tabgan_objective, n_trials=10)
    elif args.model == 'forgan':
        study.optimize(forgan_objective, n_trials=10)
    elif args.model == 'rcgan':
        study.optimize(rcgan_objective, n_trials=10)
    print('Best trial:')
    trial = study.best_trial
    print('  Value: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
    
    with open(f'{model_path}/bayes_args.pkl', 'wb') as f:
        pickle.dump(trial.params, f)
        
    end_time = datetime.now()
    optim_time = (end_time - start_time).seconds
    print(f'optimization time: {optim_time//3600}:{(optim_time%3600)//60}:{optim_time%60}')