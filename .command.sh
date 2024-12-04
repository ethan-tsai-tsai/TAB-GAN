# trial
nohup python -u strategy.py --stock 1476 --cuda 3 > ./out/1476.out 2>&1 &
nohup python -u strategy.py --stock 2330 --cuda 4 > ./out/2330.out 2>&1 &
nohup python -u strategy.py --stock 2731 --cuda 5 > ./out/2731.out 2>&1 &
nohup python -u strategy.py --stock 3008 --cuda 6 > ./out/3008.out 2>&1 &
nohup python -u strategy.py --stock 3167 --cuda 7 > ./out/3167.out 2>&1 &

# train
nohup python -u train.py --stock 1476 --cuda 3 > ./out/1476.out 2>&1 &
nohup python -u train.py --stock 2330 --cuda 4 > ./out/2330.out 2>&1 &
nohup python -u train.py --stock 2731 --cuda 5 > ./out/2731.out 2>&1 &
nohup python -u train.py --stock 3008 --cuda 6 > ./out/3008.out 2>&1 &
nohup python -u train.py --stock 3167 --cuda 7 > ./out/3167.out 2>&1 &

# bayes optimization
nohup python -u eval_simulated.py --stock 1476_simulated --name trial_1 --pred_times 100 --cuda 3 > ./out/1476.out 2>&1 &
nohup python -u eval_simulated.py --stock 2330_simulated --name trial_1 --pred_times 100 --cuda 4 > ./out/2330.out 2>&1 &
nohup python -u eval_simulated.py --stock 2731_simulated --name trial_1 --pred_times 100 --cuda 5 > ./out/2731.out 2>&1 &
nohup python -u eval_simulated.py --stock 3008_simulated --name trial_1 --pred_times 100 --cuda 6 > ./out/3008.out 2>&1 &
nohup python -u eval_simulated.py --stock 3167_simulated --name trial_1 --pred_times 100 --cuda 7 > ./out/3167.out 2>&1 &
