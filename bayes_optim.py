from torch.utils.data import DataLoader, Subset
import optuna
import utils
from arguments import *
from preprocessing import *
from train import *

def objective(trial):
    
    # set hyperparameters
    args = parse_args()
    args.mode = 'optim' # optim mode
    # data
    args.noise_dim = trial.suggest_categorical('noise_dim', [8, 16, 32, 64])
    # model
    args.hidden_dim_g = trial.suggest_categorical('hidden_dim_g', [32, 64, 128])
    args.num_layers_g = trial.suggest_categorical('num_layers_g', [1, 2, 3])
    args.hidden_dim_d = trial.suggest_categorical('hidden_dim_d', [4, 8, 16, 32, 64])
    args.num_layers_d = trial.suggest_categorical('num_layers_d', [4, 8, 16])
    # train
    args.lr_g = trial.suggest_float('lr_g', 1e-7, 1e-4, log=True)
    args.lr_d = trial.suggest_float('lr_d', 1e-7, 1e-4, log=True)
    args.epoch = trial.suggest_int('epoch', 10, 50)
    args.batch_size = trial.suggest_categorical('batch_size', [128, 256, 512, 1024])
    args.d_iter = trial.suggest_int('d_iter', 1, 5)
    
    # prepare dataset
    stock_data = StockDataset(args, mode='optim')
    train_size = int(0.9 * len(stock_data))
    test_size = len(stock_data) - train_size
    train_datasets, test_datasets = random_split(stock_data, [train_size, test_size])
    train_loader = DataLoader(train_datasets, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_datasets, batch_size=args.batch_size, shuffle=False)
    val_dates = random.sample(stock_data.time_intervals, args.num_val)
    
    print(f'Starting trial with params: {trial.params}')
    # model setup
    wgan_model = wgan(stock_data, args)
    _ = wgan_model.train(train_loader, test_loader)
    
    score = wgan_model.BEST_KLD
    return score
    

if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)
    print('Best trial:')
    trial = study.best_trial
    print('  Value: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))