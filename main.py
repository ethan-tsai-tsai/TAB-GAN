import subprocess
from arguments import *

# set parameters
args = parse_args()

# 定義腳本列表
scripts = ["preprocessor.py", "bayes_optim.py", "train.py", "eval.py"]

# 逐一執行腳本並即時打印輸出
for script in scripts:
    print(f"正在即時執行 {script} ...")
    with subprocess.Popen(['python', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1) as proc:
        for line in proc.stdout:
            print(line, end='')

        # 即時打印標準錯誤
        for err_line in proc.stderr:
            print(err_line, end='')

    print(f"{script} 執行完成。\n")
