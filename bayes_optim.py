import argparse
import optuna

from forgan import ForGAN
from arguments import *
import utils

def objective(trial):
    # 設定實驗超參數
    args = parse_args()
    args.mode = 'optim' # 切換到 optim 模式
    # data
    args.noise_dim = trial.suggest_categorical('noise_dim', [8, 16, 32, 64])
    # model
    args.hidden_dim_g = trial.suggest_categorical('hidden_dim_g', [32, 64, 128])
    args.num_layers_g = trial.suggest_categorical('num_layers_g', [1, 2, 3])
    args.hidden_dim_d = trial.suggest_categorical('hidden_dim_d', [4, 8, 16, 32, 64])
    args.num_layers_d = trial.suggest_categorical('num_layers_d', [4, 8, 16])
    # train
    args.lr_g = trial.suggest_float('lr_g', 1e-5, 1e-2, log=True)
    args.lr_d = trial.suggest_float('lr_d', 1e-5, 1e-2, log=True)
    args.epoch = trial.suggest_int('epoch', 10, 1000)
    args.batch_size = trial.suggest_categorical('batch_size', [512, 1024, 2048])
    args.d_iter = trial.suggest_int('d_iter', 1, 5)
    
    # 建立資料
    _, x_train, y_train, x_val, y_val, x_test, y_test = utils.prepare_dataset(args)
    args.data_mean = x_train.mean() # 取的 training data 的 avg 和 std
    args.data_std = x_train.std()
    
    forgan = ForGAN(args)
    forgan.train(x_train, y_train, x_val, y_val)
    rmse = forgan.test(x_test, y_test)
    return rmse
    

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.argsimize(objective, n_trials=100)
    print('Best trial:')
    trial = study.best_trial
    print('  Value: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))