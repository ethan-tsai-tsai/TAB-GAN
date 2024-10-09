from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'optim', 'test'], default='train', help='processing mode')
    # Data
    parser.add_argument('--name', type=str, default='model', help='模型名稱')
    parser.add_argument('--stock', type=str, default='0050', help='股票編號')
    parser.add_argument('--window_size', type=int, default=3, help='移動窗格大小（用幾天的資料量來預測下一天）')
    parser.add_argument('--window_stride', type=int, default=1, help='移動窗格的移動步伐')
    parser.add_argument('--target_length', type=int, default=270, help='生成器輸出的序列長度')
    parser.add_argument('--noise_dim', type=int, default=32, help='輸入生成器的噪聲維度')
    parser.add_argument('--time_step', type=int, default=30, help="輸入為每N分鐘的資料")
    parser.add_argument('--num_val', type=int, default=2, help='每次驗證需要的筆數')
    
    # Model
    parser.add_argument('--hidden_dim_g', type=int, default=64, help='generator each hidden layer dimension')
    parser.add_argument('--num_layers_g', type=int, default=1, help='generator hidden layer number')
    parser.add_argument('--hidden_dim_d', type=int, default=8, help='discriminator each hidden layer dimension')
    parser.add_argument('--num_layers_d', type=int, default=4, help='discriminator hidden layer number')
    
    # Train
    parser.add_argument('--lr_g', type=float, default=0.00005, help='generator learning rate')
    parser.add_argument('--lr_d', type=float, default=0.00001, help='discriminator learning rate')
    parser.add_argument('--cuda', type=int, default=2, help='cuda number')
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--d_iter', type=int, default=1, help='訓練幾次判別器後再訓練生成器')
    parser.add_argument('--gp_lambda', type=float, default=10, help='lambda of gradient penalty')
    
    # Evaluation
    parser.add_argument('--num_days', type=int, default=10, help='單張折線圖中的天數')
    parser.add_argument('--num_eval', type=int , default=10, help='要評估的折線圖數量')
    parser.add_argument('--pred_times', type=int, default=10, help='重複預測的次數')
    parser.add_argument('--bound_percent', type=float, default=90, help='bound 的上下界比例')
    
    args = parser.parse_args()
    
    return args