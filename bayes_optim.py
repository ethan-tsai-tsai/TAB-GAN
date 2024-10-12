from torch.utils.data import DataLoader, Subset
import optuna
import logging
import random
import utils
from arguments import *
from preprocessor import *
from train import *

optuna.logging.set_verbosity(optuna.logging.DEBUG)
def objective(trial):
    try:
        # set hyperparameters
        args = parse_args()
        args.mode = 'optim' # optim mode
        # data
        args.noise_dim = trial.suggest_categorical('noise_dim', [32, 64, 128])
        # model
        args.hidden_dim_g = trial.suggest_categorical('hidden_dim_g', [32, 64, 128])
        args.num_layers_g = trial.suggest_categorical('num_layers_g', [1, 2, 3])
        args.hidden_dim_d = trial.suggest_categorical('hidden_dim_d', [8, 16, 32])
        args.num_layers_d = trial.suggest_categorical('num_layers_d', [4, 8, 16])
        # train
        args.lr_d = trial.suggest_float('lr_d', 1e-6, 1e-4, log=True)
        args.lr_g = trial.suggest_float('lr_g', 1e-6, 1e-4, log=True)
        args.batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
        args.d_iter = trial.suggest_int('d_iter', 2, 5)
        args.gp_lambda = trial.suggest_int('gp_lambda', 6, 10)
        
        # prepare dataset
        train_datasets = StockDataset(args, f'./data/{args.stock}/train.csv')
        train_size = int(0.9 * len(train_datasets))
        val_size = len(train_datasets) - train_size
        train_data, val_data = random_split(train_datasets, [train_size, val_size])
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
        
        print(f'Starting trial with params: {trial.params}')
        # model setup
        wgan_model = wgan(train_datasets, args)
        _ = wgan_model.train(train_loader, val_loader)
        
        score = wgan_model.BEST_KLD
        return score
    except Exception as e:
        logging.error(f"Error in trial {trial.number}: {e}")
        raise
        
    

if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)
    print('Best trial:')
    trial = study.best_trial
    print('  Value: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
        print(f'--{key} {value}\\')