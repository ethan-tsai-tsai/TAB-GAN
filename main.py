# Import packages
import sys
import pandas as pd
from subprocess import Popen, PIPE
# Import files
from arguments import parse_args
from lib.data import DataProcessor

def verify_data(args, trial):
    """驗證當前 trial 的資料是否正確"""
    train_data = pd.read_csv(f'./data/{args.stock}/train.csv', index_col='ts')
    test_data = pd.read_csv(f'./data/{args.stock}/test.csv', index_col='ts')
    
    print(f"\nTrial {trial} 資料驗證:")
    print(f"訓練資料範圍: {train_data.index[0]} to {train_data.index[-1]}")
    print(f"測試資料範圍: {test_data.index[0]} to {test_data.index[-1]}")

def run_trial(args, trial):
    print(f"\n{'='*50}")
    print(f"Trial {trial}")
    print(f"{'='*50}\n")
    scripts = ["bayes_optim.py", "train.py", "eval.py"]
   
   # get trial data
    _ = DataProcessor(args, trial)
    verify_data(args, trial)
    
    # set up variables
    args.name = f'trial_{trial}'

    for script in scripts:
        print(f"執行 {script}")
        try:
            process = Popen(
                [sys.executable, script,
                 '--model', args.model,
                 '--stock', args.stock,
                 '--name', args.name,
                 '--cuda', str(args.cuda)],
                stdout=PIPE,
                stderr=PIPE,
                bufsize=1,
                universal_newlines=True
            )
            while True:
                output = process.stdout.readline()
                error = process.stderr.readline()
                if output: print(output.strip())
                if error: print(f"錯誤：{error.strip()}")
                if output == '' and error == '' and process.poll() is not None:
                    break
            if process.returncode != 0:
                print(f"\n錯誤：{script} 執行失敗，錯誤碼：{process.returncode}")
                sys.exit(1)
                
        except FileNotFoundError:
            print(f"\n錯誤：找不到檔案 {script}")
            sys.exit(1)

if __name__ == "__main__":
    args = parse_args()
    args.mode = 'test'
    print(f'fitting model: {args.model}')
    run_trial(args, trial=1)