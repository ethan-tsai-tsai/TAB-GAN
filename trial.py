# Import packages
import sys
from subprocess import Popen, PIPE
# Import files
from arguments import parse_args
from preprocessor import *

def run_trial(args, trial):
    print(f"\n{'='*50}")
    print(f"Trial {trial}")
    print(f"{'='*50}\n")
    scripts = ["bayes_optim.py", "train.py", "eval.py"]
   
   # get trial data
    _ = DataProcessor(args, trial)
    
    # set up variables
    args.name = f'trial_{trial}'

    for script in scripts:
        print(f"執行 {script}")
        
        try:
            # 使用 Popen 即時顯示輸出
            process = Popen(
                [sys.executable, script, '--stock', args.stock, '--name', args.name, '--cuda', str(args.cuda)],
                stdout=PIPE,
                stderr=PIPE,
                bufsize=1,
                universal_newlines=True
            )
            
            # 即時讀取並輸出
            while True:
                output = process.stdout.readline()
                error = process.stderr.readline()
                
                # 輸出標準輸出
                if output:
                    print(output.strip())
                
                # 輸出錯誤訊息
                if error:
                    print(f"錯誤：{error.strip()}")
                
                # 檢查程式是否執行完畢
                if output == '' and error == '' and process.poll() is not None:
                    break
            
            # 檢查返回碼
            if process.returncode != 0:
                print(f"\n錯誤：{script} 執行失敗，錯誤碼：{process.returncode}")
                sys.exit(1)
                
        except FileNotFoundError:
            print(f"\n錯誤：找不到檔案 {script}")
            sys.exit(1)

if __name__ == "__main__":
    args = parse_args()
    num_trials = 36
    for i in range(1, num_trials + 1):
        run_trial(args, i)