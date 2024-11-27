# trial
nohup python -u strategy.py --stock 1476_simulated --cuda 3 > ./out/1476.out 2>&1 &
nohup python -u strategy.py --stock 2330_simulated --cuda 4 > ./out/2330.out 2>&1 &
nohup python -u strategy.py --stock 2731_simulated --cuda 5 > ./out/2731.out 2>&1 &
nohup python -u strategy.py --stock 3008_simulated --cuda 6 > ./out/3008.out 2>&1 &
nohup python -u strategy.py --stock 3167_simulated --cuda 7 > ./out/3167.out 2>&1 &

# train
nohup python -u train.py --stock 1476 --cuda 3 > ./out/1476.out 2>&1 &
nohup python -u train.py --stock 2330 --cuda 4 > ./out/2330.out 2>&1 &
nohup python -u train.py --stock 2731 --cuda 5 > ./out/2731.out 2>&1 &
nohup python -u train.py --stock 3008 --cuda 6 > ./out/3008.out 2>&1 &
nohup python -u train.py --stock 3167 --cuda 7 > ./out/3167.out 2>&1 &

# bayes optimization
nohup python -u bayes_optim.py --stock 1476 --cuda 3 > ./out/optim_1476.out 2>&1 &
nohup python -u bayes_optim.py --stock 2330 --cuda 4 > ./out/optim_2330.out 2>&1 &
nohup python -u bayes_optim.py --stock 2731 --cuda 5 > ./out/optim_2731.out 2>&1 &
nohup python -u bayes_optim.py --stock 3008 --cuda 6 > ./out/optim_3008.out 2>&1 &
nohup python -u bayes_optim.py --stock 3167 --cuda 7 > ./out/optim_3167.out 2>&1 &