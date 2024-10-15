# train
nohup python -u train.py --stock 0050 --cuda 4 --epoch 100 --hidden_dim_d 256 --num_layers_d 4 > ./out/0050.out 2>&1 &
nohup python -u train.py --stock 1101 --cuda 5 --epoch 100 --hidden_dim_d 256 --num_layers_d 4 > ./out/1101.out 2>&1 &
nohup python -u train.py --stock 2330 --cuda 6 --epoch 100 --hidden_dim_d 256 --num_layers_d 4 > ./out/2330.out 2>&1 &
nohup python -u train.py --stock 2610 --cuda 7 --epoch 100 --hidden_dim_d 256 --num_layers_d 4 > ./out/2610.out 2>&1 &
## 0050
nohup python -u train.py > ./out/0050.out 2>&1 &
## 1101
nohup python -u train.py \
    --stock 1101\
    --cuda 5\
    --noise_dim 32\
    --hidden_dim_g 128\
    --num_layers_g 2\
    --hidden_dim_d 8\
    --num_layers_d 4\
    --lr_d 3.122685523175086e-05\
    --lr_g 2.8915138363367947e-05\
    --batch_size 128\
    --d_iter 2\
    --gp_lambda 10\
    --epoch 300\
    > ./out/1101.out 2>&1 &
## 2330
nohup python -u train.py \
    --stock 2330\
    --cuda 6\
    --noise_dim 64\
    --hidden_dim_g 128\
    --num_layers_g 1\
    --hidden_dim_d 32\
    --num_layers_d 8\
    --lr_d 3.2386357487889878e-06\
    --lr_g 5.113940871925737e-06\
    --batch_size 128\
    --d_iter 5\
    --gp_lambda 9\
    --epoch 300\
    > ./out/2330.out 2>&1 &
## 2610
nohup python -u train.py --stock 2610 --cuda 7&

# bayes optimization
nohup python -u bayes_optim.py --stock 0050 --cuda 6 > ./out/optim_0050.out 2>&1 &
nohup python -u bayes_optim.py --stock 1101 --cuda 3 > ./out/optim_1101.out 2>&1 &
nohup python -u bayes_optim.py --stock 2330 --cuda 7 > ./out/optim_2330.out 2>&1 &
nohup python -u bayes_optim.py --stock 2610 --cuda 4 > ./out/optim_2610.out 2>&1 &