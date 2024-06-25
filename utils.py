import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import random
from datetime import datetime, timedelta

def save_loss_curve(results, args):
    print('Saving loss curve')
    plt.figure(figsize=(20, 10))
    # train loss curve
    plt.subplot(1, 2, 1)
    plt.axhline(y=0, color='gray', linestyle='-')
    plt.plot(range(args.epoch), results['loss_d'], label='Discriminator Loss')
    plt.plot(range(args.epoch), results['loss_g'], label='Generator Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # test loss curve
    plt.subplot(1, 2, 2)
    plt.axhline(y=0, color='gray', linestyle='-')
    plt.plot(range(args.epoch), results['test_loss_d'], label='Discriminator Loss')
    plt.plot(range(args.epoch), results['test_loss_g'], label='Generator Loss')
    plt.title('Testing Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'./img/loss/{args.stock}_loss_{args.name}.png')
    
def save_model(model_d, model_g, args):
    print(f"Saving model")
    torch.save({
        'args': args,
        'model_d': model_d.state_dict(),
        'model_g': model_g.state_dict()
    }, f'./model/{args.stock}_{args.name}.pth')

def save_predict_plot(args, dates, y_preds, y_true=None):
    # 設定變數
    seq_unit = 270//args.time_step
    target_length = args.target_length//args.time_step
    
    # 取得時間陣列
    start_time = datetime.strptime('09:00', '%H:%M')
    end_time = start_time + timedelta(minutes=args.target_length)
    stock_end_time = datetime.strptime('13:30', '%H:%M')
    x_ticks_interval = 45

    pre_time_array = []
    time_array = []
    current_time = start_time
    while current_time < stock_end_time:
        if current_time < end_time: time_array.append(current_time.strftime('%H:%M'))
        pre_time_array.append(current_time.strftime('%H:%M'))
        current_time += timedelta(minutes=x_ticks_interval)
    time_array.append(current_time.strftime('%H:%M'))
    
    # 設定x座標
    x_labels = pre_time_array * (args.num_days-1) + time_array
    x_ticks = [i * x_ticks_interval/args.time_step + 1 for i in range(len(x_labels))]
    colors = list(mpl.colors.cnames)
    
    # plot
    plt.figure(figsize=(20, 10))
    plt.grid(True)
    for i in range(len(y_preds)):
        plt.plot(range(i, i+target_length), y_preds[i], color=colors[6+i])
    if y_true is not None:
        plt.plot(range(seq_unit*(args.num_days-1)+target_length), y_true, color='black', label='real')
    plt.legend()
    plt.xticks(x_ticks, x_labels, rotation=15)
    plt.title(f'{dates[0]} - {dates[-1]}')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.savefig(f'./img/pred/{args.stock}_{args.name}/pred_{dates[-1]}.png')
    print(f'Svaving prediction: {dates[-1]}')
    plt.close()

def clear_folder(folder_path):
    # 確保資料夾存在
    if not os.path.exists(folder_path):
        print(f"資料夾 {folder_path} 不存在。")
        return
    
    # 遍歷資料夾中的所有檔案
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            # 如果是檔案，則刪除
            if os.path.isfile(file_path):
                os.remove(file_path)
            # 如果是資料夾，則遞迴清空
            elif os.path.isdir(file_path):
                clear_folder(file_path)
        except Exception as e:
            print(f"刪除 {file_path} 時發生錯誤: {e}")