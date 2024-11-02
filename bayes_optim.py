from torch.utils.data import DataLoader
import optuna
import logging
import pickle
from arguments import *
from preprocessor import *
from train import *

optuna.logging.set_verbosity(optuna.logging.DEBUG)
def objective(trial):
    try:
        # set hyperparameters
        args.mode = 'optim' # optim mode
        # data
        args.noise_dim = trial.suggest_categorical('noise_dim', [32, 64, 128, 256])
        # model
        args.hidden_dim_g = trial.suggest_categorical('hidden_dim_g', [32, 64, 128])
        args.num_layers_g = trial.suggest_int('num_layers_g', 1, 4)
        args.num_head_g = trial.suggest_categorical('num_head_g', [4, 8, 16])
        args.hidden_dim_d = trial.suggest_categorical('hidden_dim_d', [16, 32, 64, 128])
        args.num_layers_d = trial.suggest_int('num_layers_d', 1, 4)
        args.num_head_d = trial.suggest_categorical('num_head_d', [4, 8, 16])
        # train
        args.epoch  = trial.suggest_int('epoch', 10, 200)
        args.lr_d = trial.suggest_float('lr_d', 1e-5, 1e-4, log=True)
        args.lr_g = trial.suggest_float('lr_g', 1e-5, 1e-4, log=True)
        args.batch_size = trial.suggest_categorical('batch_size', [128, 256, 512, 1024])
        args.d_iter = trial.suggest_int('d_iter', 1, 5)
        args.gp_lambda = trial.suggest_int('gp_lambda', 1, 10)
        
        # prepare dataset
        train_datasets = StockDataset(args, f'./data/{args.stock}/train.csv')
        test_datasets = StockDataset(args, f'./data/{args.stock}/test.csv')
        train_size = int(0.95 * len(train_datasets))
        val_size = len(train_datasets) - train_size
        train_data, val_data = random_split(train_datasets, [train_size, val_size])
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_datasets, batch_size=args.batch_size, shuffle=False)
        
        print(f'Starting trial with params: {trial.params}')
        # model setup
        wgan_model = wgan(train_datasets, args)
        _ = wgan_model.train(train_loader, val_loader)
        _, _, test_kld, test_fid = wgan_model.validation(test_loader)
        _, _, val_kld, val_fid = wgan_model.validation(val_loader)
        test_score = test_kld + test_fid
        val_score = val_kld + val_fid
        alpha = 0.7
        score = alpha * test_score + (1 - alpha) * val_score
        return score
    except Exception as e:
        logging.error(f"Error in trial {trial.number}: {e}")
        raise
        
    

if __name__ == '__main__':
    
    start_time = datetime.now()
    
    args = parse_args()
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)
    print('Best trial:')
    trial = study.best_trial
    print('  Value: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
    
    with open(f'./model/{args.stock}_{args.name}/bayes_args.pkl', 'wb') as f:
        pickle.dump(trial.params, f)
        
    end_time = datetime.now()
    optim_time = (end_time - start_time).seconds
    print(f'optimization time: {optim_time//3600}:{(optim_time%3600)//60}:{optim_time%60}')