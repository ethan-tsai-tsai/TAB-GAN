import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import os
import numpy as np
from datetime import datetime, timedelta, time
from random import choice

def price_change(data):
    changes = [0]  # 第一天沒有變化，設為 0
    for i in range(1, len(data)):
        if data[i - 1] == 0:  # 檢查前一天的價格是否為零
            changes.append(0)  # 如果為零，無法計算變化，設為 0 或其他值
        else:
            change = (data[i] - data[i - 1]) / data[i - 1] * 100
            changes.append(change)
    return changes

def calc_kld(generated_data, ground_truth, bins, range_min, range_max):
    pd_gt, _ = np.histogram(ground_truth, bins=bins, density=True, range=(range_min, range_max))
    pd_gen, _ = np.histogram(generated_data, bins=bins, density=True, range=(range_min, range_max))
    kld = 0
    for x1, x2 in zip(pd_gt, pd_gen):
        if x1 != 0 and x2 == 0:
            kld += x1
        elif x1 == 0 and x2 != 0:
            kld += x2
        elif x1 != 0 and x2 != 0:
            kld += x1 * np.log(x1 / x2)

    return np.abs(kld)

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

def save_predict_plot(args, path, file_name, dates, y_preds, y_true=None):
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
    colors = list(mpl.colors.cnames) * 10
    
    # plot
    plt.figure(figsize=(20, 10))
    plt.grid(True)
    for i in range(len(y_preds)//args.pred_times):
        x_vals = range(i*args.window_stride, i*args.window_stride + target_length)
        color = choice(colors)
        # get upper and lower bound
        upper_bound = []
        lower_bound = []
        for j in zip(*y_preds[args.pred_times * i: args.pred_times * (i + 1)]):
            upper_bound.append(np.percentile(j, args.bound_percent))
            lower_bound.append(np.percentile(j, 100 - args.bound_percent))
        
        plt.fill_between(x_vals, lower_bound, upper_bound, color=color, alpha=0.5)
        for j in range(i*args.pred_times, args.pred_times * (i+1)):
            plt.scatter(x_vals, y_preds[j], color=color, alpha=0.3)
            if y_true is not None:
                plt.plot(x_vals, y_true[j], color='black', label='real')
    # plt.legend()
    # plt.xticks(x_ticks, x_labels, rotation=15)
    plt.title(f'{dates[0]} - {dates[-1]}')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.savefig(f'{path}/{file_name}.png')
    plt.close()

def save_dist_plot(args, path, file_name, dates, y_preds, y_true):
    y_true = np.array(y_true).flatten()
    y_preds = np.array(y_preds).flatten()
    target_length = args.target_length//args.time_step
    n_plot = math.ceil(math.sqrt(target_length)) # 圖片一欄/列要幾張
    start_time = datetime.combine(datetime.today(), time(9,0))
    _, axes = plt.subplots(n_plot, n_plot, figsize=(10,10))
    idx=0
    for row in range(n_plot):
        for col in range(n_plot):
            axes[row, col].hist(y_preds[idx::target_length], bins='auto', density=True, alpha=0.5, color='green')
            axes[row, col].hist(y_true[idx::target_length], bins='auto', density=True, alpha=0.5, color='red')
            axes[row, col].set_title(f'{(start_time + timedelta(minutes=args.time_step*(idx+1))).strftime("%H:%M")}')
            idx += 1
    plt.tight_layout()
    plt.savefig(f'{path}/{file_name}.png')
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