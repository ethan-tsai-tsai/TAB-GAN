# train
nohup python -u train.py --stock 0050 --cuda 4 > ./out/0050.out 2>&1 &
nohup python -u train.py --stock 1101 --cuda 5 > ./out/1101.out 2>&1 &
nohup python -u train.py --stock 2330 --cuda 6 > ./out/2330.out 2>&1 &
nohup python -u train.py --stock 2610 --cuda 7 > ./out/2610.out 2>&1 &

# bayes optimization
nohup python -u bayes_optim.py --stock 0050 --cuda 6 > ./out/optim_0050.out 2>&1 &
nohup python -u bayes_optim.py --stock 1101 --cuda 3 > ./out/optim_1101.out 2>&1 &
nohup python -u bayes_optim.py --stock 2330 --cuda 7 > ./out/optim_2330.out 2>&1 &
nohup python -u bayes_optim.py --stock 2610 --cuda 4 > ./out/optim_2610.out 2>&1 &

# evaluation
python -u eval.py --stock 0050 --cuda 4
python -u eval.py --stock 1101 --cuda 5
python -u eval.py --stock 2330 --cuda 6
python -u eval.py --stock 2610 --cuda 7