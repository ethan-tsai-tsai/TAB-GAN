import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from datetime import datetime, timedelta

def calc_kld(generated_data, ground_truth):
    pd_gt, _ = np.histogram(ground_truth, bins='auto', density=True)
    pd_gen, _ = np.histogram(generated_data, bins='auto', density=True)
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
                os.rmdir(file_path)
        except Exception as e:
            print(f"刪除 {file_path} 時發生錯誤: {e}")

class plot_predicions:
    def __init__(self, path, args, time_interval):
        self.args = args
        self.path = path
        self.seq_len = 270 // self.args.time_step
        self.target_len = self.args.target_length // self.args.time_step
        self.time_array = self._get_time_interval()
        self.time_interval = time_interval
        # check folder exists
        if not os.path.exists(self.path): os.makedirs(path)
        else: clear_folder(self.path)
    
    def dist_plot(self, y_true, y_pred):
        # processse array
        arrange_y_pred = y_pred.transpose(1, 0, 2).reshape(y_pred.shape[1], y_pred.shape[0] * y_pred.shape[2])
        arrange_y_true = y_true.T

        file_name = f'{self.time_interval[0]} - {self.time_interval[-1]}'
        _, axes = plt.subplots(3, 3, figsize=(12, 12))
        for i, ax in enumerate(axes.flat):
            sns.histplot(arrange_y_pred[i], ax=ax, stat='density', color='green', alpha=0.3)
            if len(arrange_y_true[i]) == 1:
                ax.axvline(x=arrange_y_true[i], color='black', linestyle='--', linewidth=5)
            else:
                sns.histplot(arrange_y_true[i], ax=ax, stat='density', color='red', alpha=0.3)
            ax.set_title(f'{self.time_array[i]}')
        plt.tight_layout()
        plt.savefig(f'{self.path}/{file_name}_dist.png')
        plt.close()
    
    def band_plot(self, y_true, y_pred):
        file_name = f'{self.time_interval[0]} - {self.time_interval[-1]}'
        palette = sns.color_palette('pastel', len(self.time_interval) * self.seq_len)
        plt.figure(figsize=(10, 5))
        for i, (y, y_hat) in enumerate(zip(y_true, y_pred)):
            # calculate upper and lower bound
            upper_bound, lower_bound = self._get_bound(y_hat)
            # set values
            y_hat = np.array(y_hat).flatten()
            x_val = np.arange(i, i + self.seq_len)
            # ploting
            sns.lineplot(x=x_val, y=y, color='black')
            sns.scatterplot(x=x_val.repeat(self.args.pred_times), y=y_hat, alpha=0.2, color=palette[i])
            plt.fill_between(x_val, lower_bound, upper_bound, color='gray', alpha=0.8, zorder=10)
            
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.xticks(ticks=range(0, y_pred.shape[0], self.seq_len), labels=[self.time_array[0]] * len(self.time_interval))
        plt.legend().remove()
        plt.savefig(f'{self.path}/{file_name}_bound.png')
        plt.close()
    
    def fixed_band_plot(self, y_true, y_pred):
        file_name = f'fixed_{self.time_interval[0]} - {self.time_interval[-1]}'
        palette = sns.color_palette('pastel', len(self.time_interval) * self.seq_len)
        plt.figure(figsize=(10, 5))
        for i, (y, y_hat) in enumerate(zip(y_true, y_pred)):
            # calculate upper and lower bound
            upper_bound, lower_bound = self._get_bound(y_hat)
            # set values
            y_hat = np.array(y_hat).flatten()
            x_val = np.arange(i, i + self.seq_len)
            # ploting
            sns.lineplot(x=x_val, y=y, color='black')
            sns.scatterplot(x=x_val.repeat(self.args.pred_times), y=y_hat, alpha=0.2, color=palette[i])
            plt.fill_between(x_val, lower_bound, upper_bound, color='gray', alpha=0.8, zorder=10)
            
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.xticks(ticks=range(0, y_pred.shape[0], self.seq_len), labels=[self.time_array[0]] * len(self.time_interval))
        plt.legend().remove()
        plt.savefig(f'{self.path}/{file_name}_bound.png')
        plt.close()
    
    def single_time_plot(self, y_true, y_pred):
        """
        plot single time prediction with scatter plot and histogram
        """
        palette = sns.color_palette('pastel', self.seq_len)
        for i, date in enumerate(self.time_interval):
            if not os.path.exists(f'{self.path}/{date}'): os.mkdir(f'{self.path}/{date}')
            else: clear_folder(f'{self.path}/{date}')
            # create axes
            fig = plt.figure(figsize=(10, 5))
            gs = mpl.gridspec.GridSpec(1, 2, width_ratios=[3, 1])
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            x_val = np.arange(self.seq_len)
            # plot unchanged parts
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Price')
            ax1.set_xticks(x_val)
            ax1.set_xticklabels(self.time_array)
            ax2.set_xlabel('Price')
            ax2.set_ylabel('P(Price)')
            
            # set values
            y = np.array(y_true[i*self.seq_len])
            y_hat = []
            for idx, sublist in enumerate(y_pred[i * self.seq_len: (i + 1) * self.seq_len]):
                if idx == 0: y_hat.append(sublist)
                else: y_hat.append(sublist[:-idx:, :])
            
            # band plot
            sns.lineplot(x=x_val, y=y, color='black', ax=ax1)
            for j in range(self.seq_len):
                upper_bound, lower_bound = self._get_bound(y_hat[j])
                x_val = np.arange(j, self.seq_len)
                sns.scatterplot(x=x_val.repeat(self.args.pred_times), y=y_hat[j].flatten(), color=palette[j], ax=ax1, alpha=0.3)
                ax1.fill_between(x_val, lower_bound, upper_bound, color=palette[j], alpha=0.7)
                # histogram
                ax2.cla()
                sns.histplot(x=y_hat[j][0], stat='density', color='blue', alpha=0.3, ax=ax2)
                ax2.axvline(x=y_true[i*self.seq_len][j], color='black', linestyle='--', linewidth=5)
                
                # # save fig
                ax1.set_title(f'{date} - {self.time_array[j]}')
                plt.savefig(f'{self.path}/{date}/{self.time_array[j]}.png')
            plt.close()
        
    def _get_time_interval(self):
        # 取得時間陣列
        start_time = datetime.strptime('09:00', '%H:%M')
        end_time = datetime.strptime('13:30', '%H:%M')
        x_ticks_interval = self.args.time_step

        time_array = []
        time_array = []
        current_time = start_time
        while current_time < end_time:
            current_time += timedelta(minutes=x_ticks_interval)
            time_array.append(current_time.strftime('%H:%M'))
        return time_array
        
    def _get_bound(self, y_preds):
        upper_bound = []
        lower_bound = []
        
        for pred in y_preds:
            upper_bound.append(np.percentile(pred, self.args.bound_percent))
            lower_bound.append(np.percentile(pred, 100 - self.args.bound_percent))

        return np.array(upper_bound), np.array(lower_bound)