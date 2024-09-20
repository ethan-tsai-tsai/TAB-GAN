# train
nohup python -u train.py --stock 0050 --cuda 4 --epoch 50 > ./out/0050.out 2>&1 &
nohup python -u train.py --stock 1101 --cuda 5 --epoch 50 > ./out/1101.out 2>&1 &
nohup python -u train.py --stock 2330 --cuda 6 --epoch 50 > ./out/2330.out 2>&1 &
nohup python -u train.py --stock 2610 --cuda 7 --epoch 50 > ./out/2610.out 2>&1 &

# evaluation
python eval.py --stock 0050 --cuda 0 &
python eval.py --stock 1101 --cuda 1 &
python eval.py --stock 2330 --cuda 2 &
python eval.py --stock 2610 --cuda 3 &

# bayes optimization
nohup python -u bayes_optim.py --stock 0050 --cuda 1 > ./out/0050_optim.out 2>&1 &
nohup python -u bayes_optim.py --stock 1101 --cuda 2 > ./out/1101_optim.out 2>&1 &
nohup python -u bayes_optim.py --stock 2330 --cuda 3 > ./out/2330_optim.out 2>&1 &
nohup python -u bayes_optim.py --stock 2610 --cuda 4 > ./out/2610_optim.out 2>&1 &