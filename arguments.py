from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'optim', 'test'], default='train', help='processing mode')
    # Data
    parser.add_argument('--name', type=str, default='model', help='模型名稱')
    parser.add_argument('--stock', type=str, default='2330', help='股票編號')
    parser.add_argument('--window_size', type=int, default=3, help='移動窗格大小（用幾天的資料量來預測下一天）')
    parser.add_argument('--window_stride', type=int, default=1, help='移動窗格的移動步伐')
    parser.add_argument('--target_length', type=int, default=270, help='生成器輸出的序列長度')
    parser.add_argument('--time_step', type=int, default=10, help="輸入為每N分鐘的資料")
    
    # Model
    parser.add_argument('--hidden_dim_g', type=int, default=128, help='generator each hidden layer dimension')
    parser.add_argument('--num_layers_g', type=int, default=1, help='generator hidden layer number')
    parser.add_argument('--num_head_g', type=int, default=8, help='self attention head')
    parser.add_argument('--hidden_dim_d', type=int, default=64, help='discriminator each hidden layer dimension')
    parser.add_argument('--num_layers_d', type=int, default=3, help='discriminator hidden layer number')
    parser.add_argument('--num_head_d', type=int, default=8, help='self attention head')
    
    # Train
    parser.add_argument('--lr_g', type=float, default=0.000005, help='generator learning rate')
    parser.add_argument('--lr_d', type=float, default=0.000001, help='discriminator learning rate')
    parser.add_argument('--cuda', type=int, default=2, help='cuda number')
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs')
    parser.add_argument('--noise_dim', type=int, default=64, help='輸入生成器的噪聲維度')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--d_iter', type=int, default=4, help='iteration of training discriminator before training generator')
    parser.add_argument('--gp_lambda', type=float, default=5, help='lambda of gradient penalty')
    
    # Evaluation
    parser.add_argument('--pred_times', type=int, default=10, help='prediction time in one time point')
    parser.add_argument('--bound_percent', type=float, default=90, help='confidence of bound')
    
    args = parser.parse_args()
    
    return args