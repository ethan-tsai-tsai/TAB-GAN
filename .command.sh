# train
nohup python -u train.py --stock 0050 --cuda 4 --epoch 100 > ./out/0050.out 2>&1 &
nohup python -u train.py --stock 1101 --cuda 5 --epoch 100 > ./out/1101.out 2>&1 &
nohup python -u train.py --stock 2330 --cuda 6 --epoch 100 > ./out/2330.out 2>&1 &
## 2610
nohup python -u train.py \
    --stock 2610\
    --cuda 7\
    --noise_dim 8\
    --hidden_dim_g 32\
    --num_layers_g 1\
    --hidden_dim_d 8\
    --num_layers_d 4\
    --lr_g 2.5519988501236693e-05\
    --lr_d 3.855035071257917e-05\
    --epoch 325\
    --batch_size 1024\
    --d_iter 3\
    --gp_lambda 9\
    > ./out/2610.out 2>&1 &

# evaluation
python eval.py\
    --stock 2610\
    --cuda 7\
    --noise_dim 8\
    --hidden_dim_g 32\
    --num_layers_g 1\
    --hidden_dim_d 8\
    --num_layers_d 4\
    --lr_g 2.5519988501236693e-05\
    --lr_d 3.855035071257917e-05\
    --epoch 325\
    --batch_size 1024\
    --d_iter 3\
    --gp_lambda 9\
    --num_days 2\
    --pred_times 100\
    --bound_percent 90&
python eval.py --stock 1101 --cuda 1 &
python eval.py --stock 2330 --cuda 2 &
python eval.py --stock 2610 --cuda 3 &

# bayes optimization
nohup python -u bayes_optim.py --stock 0050 --cuda 6 > ./out/optim_0050.out 2>&1 &
nohup python -u bayes_optim.py --stock 1101 --cuda 2 > ./out/optim_1101.out 2>&1 &
nohup python -u bayes_optim.py --stock 2330 --cuda 7 > ./out/optim_2330.out 2>&1 &
nohup python -u bayes_optim.py --stock 2610 --cuda 4 > ./out/optim_2610.out 2>&1 &