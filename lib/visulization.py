# import packages
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
# import file
from lib.utils import clear_folder

def visualize_band(args):
    if not os.path.exists('./img/trading_signals'): os.makedirs('./img/trading_signals')
    file_path = f'./data/trading_signals_{args.stock}.csv'
    output_path = f'./img/trading_signals/{args.stock}.png'
    # Read data
    df = pd.read_csv(file_path, parse_dates=['ts'])
    
    # Define color scheme for different bands
    color_scheme = {
        'bollinger': {
            'color': '#2C3E50',
            'alpha': 0.3,
            'signal_color': '#34495E',
            'title': 'Bollinger Bands'
        },
        '50': {
            'color': '#E74C3C',
            'alpha': 0.3,
            'signal_color': '#C0392B',
            'title': '50% Prediction Band'
        },
        '70': {
            'color': '#2ECC71',
            'alpha': 0.3,
            'signal_color': '#27AE60',
            'title': '70% Prediction Band'
        },
        '90': {
            'color': '#9B59B6',
            'alpha': 0.3,
            'signal_color': '#8E44AD',
            'title': '90% Prediction Band'
        }
    }
    
    # Create figure and subplots with adjusted height ratios and spacing
    fig = plt.figure(figsize=(15, 25))
    gs = fig.add_gridspec(5, 1, height_ratios=[0.2, 1, 1, 1, 1], hspace=0.3)
    
    # Create title subplot and trading subplots
    title_ax = fig.add_subplot(gs[0])
    title_ax.axis('off')
    title_ax.text(0.5, 0.5, 'Trading Strategies Comparison', 
                 ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Create trading subplots
    axes = [fig.add_subplot(gs[i]) for i in range(1, 5)]
    
    sns.set_style("whitegrid")
    
    # Configure each subplot
    band_configs = [
        {
            'type': 'bollinger',
            'upper': 'bolling_upper',
            'lower': 'bolling_lower',
            'signals': 'bolling_signals'
        },
        {
            'type': '50',
            'upper': 'pred_upper_50',
            'lower': 'pred_lower_50',
            'signals': 'pred_signals_50'
        },
        {
            'type': '70',
            'upper': 'pred_upper_70',
            'lower': 'pred_lower_70',
            'signals': 'pred_signals_70'
        },
        {
            'type': '90',
            'upper': 'pred_upper_90',
            'lower': 'pred_lower_90',
            'signals': 'pred_signals_90'
        }
    ]
    
    for ax, config in zip(axes, band_configs):
        band_type = config['type']
        colors = color_scheme[band_type]
        
        # Plot price line
        ax.plot(df['ts'], df['Close'], color='black', linewidth=1.5, 
                label='Close Price', zorder=5)
        
        # Plot band
        ax.fill_between(df['ts'], 
                       df[config['upper']], 
                       df[config['lower']],
                       alpha=colors['alpha'],
                       color=colors['color'],
                       label=colors['title'])
        
        # Plot buy signals
        buy_signals = df[df[config['signals']] == 1]
        if not buy_signals.empty:
            ax.scatter(buy_signals['ts'], 
                      buy_signals['Close'] * 0.99,
                      marker='^', 
                      s=100,
                      color=colors['signal_color'],
                      label='Buy Signal',
                      zorder=6)
        
        # Plot sell signals
        sell_signals = df[df[config['signals']] == -1]
        if not sell_signals.empty:
            ax.scatter(sell_signals['ts'], 
                      sell_signals['Close'] * 1.01,
                      marker='v', 
                      s=100,
                      color=colors['signal_color'],
                      label='Sell Signal',
                      zorder=6)
        
        # Customize subplot
        ax.set_title(colors['title'], fontsize=12, pad=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time' if ax == axes[-1] else '')
        ax.set_ylabel('Price')
        
        # Format x-axis
        ax.tick_params(axis='x', rotation=0)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        
        # Add legend inside the plot
        ax.legend(loc='lower right', framealpha=0.9)
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
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
    plt.savefig(f'./img/{args.model}/{args.stock}_{args.name}/loss.png')

class plot_predicions:
    def __init__(self, path, args, time_interval):
        self.args = args
        self.path = path
        self.seq_len = 270 // self.args.time_step
        self.target_len = self.args.target_length // self.args.time_step
        self.time_array = self._get_time_interval()
        self.time_interval = time_interval
        # check folder exists
        if self.args.mode != 'trial':
            if not os.path.exists(self.path): os.makedirs(path)
            else: clear_folder(self.path)
    
    def dist_plot(self, y_true, y_pred):
        # process array
        arrange_y_pred = self._arrange_list_dist(y_pred)
        arrange_y_true = self._arrange_list_dist(np.expand_dims(y_true, axis=2))

        _, axes = plt.subplots(3, 3, figsize=(12, 12))
        for i, ax in enumerate(axes.flat):
            sns.histplot(arrange_y_pred[i], ax=ax, stat='density', color='green', alpha=0.3)
            if len(arrange_y_true[i]) == 1:
                ax.axvline(x=arrange_y_true[i], color='black', linestyle='--', linewidth=5)
            else:
                sns.histplot(arrange_y_true[i], ax=ax, stat='density', color='red', alpha=0.3)
            ax.set_title(f'{self.time_array[i]}')
        plt.tight_layout()
        plt.savefig(f'{self.path}/dist.png')
        plt.close()
    
    def dist_simulate_plot(self, y_simulated, y_pred, file_name):
        _, axes = plt.subplots(3, 3, figsize=(12, 12))
        for i, ax in enumerate(axes.flat):
            sns.histplot(y_pred[i], ax=ax, stat='density', color='green', alpha=0.3, legend=False)
            sns.histplot(y_simulated[i], ax=ax, stat='density', color='red', alpha=0.3, legend=False)
            ax.set_title(f'{self.time_array[i]}')
        plt.tight_layout()
        plt.savefig(f'{self.path}/{file_name}.png')
        plt.close()
    
    def fixed_dist_plot(self, y_true, y_pred):
        # process array
        arrange_y_pred = self._arrange_list_dist_fixed(y_pred).transpose(1, 0, 2).reshape(y_pred.shape[1], y_pred.shape[0] * y_pred.shape[2])
        arrange_y_true = self._arrange_list_dist_fixed(y_true).T

        _, axes = plt.subplots(3, 3, figsize=(12, 12))
        for i, ax in enumerate(axes.flat):
            sns.histplot(arrange_y_pred[i], ax=ax, stat='density', color='green', alpha=0.3)
            if len(arrange_y_true[i]) == 1:
                ax.axvline(x=arrange_y_true[i], color='black', linestyle='--', linewidth=5)
            else:
                sns.histplot(arrange_y_true[i], ax=ax, stat='density', color='red', alpha=0.3)
            ax.set_title(f'{self.time_array[i]}')
        plt.tight_layout()
        plt.savefig(f'{self.path}/dist_fixed.png')
        plt.close()
    
    def band_plot(self, y_true, y_pred):
        palette = sns.color_palette('pastel', len(self.time_interval) * self.seq_len * self.seq_len)
        
        plt.figure(figsize=(10, 5))
        for i, (y, y_hat) in enumerate(zip(y_true, y_pred)):
            # calculate upper and lower bound
            upper_bound, lower_bound = self.get_bound(y_hat)
            # set values
            y_hat = np.array(y_hat).flatten()
            x_val = np.arange(i, i + self.target_len)
            # ploting
            sns.lineplot(x=x_val, y=y, color='black')
            sns.scatterplot(x=x_val.repeat(self.args.pred_times), y=y_hat, alpha=0.4, color=palette[i])
            plt.fill_between(x_val, lower_bound, upper_bound, color='gray', alpha=0.8, zorder=10)
            
        plt.xlabel('Time')
        plt.ylabel('Price')
        # plt.xticks(ticks=range(0, y_pred.shape[0], self.seq_len), labels=[self.time_array[0]] * len(self.time_interval))
        plt.title(f'{self.time_interval[0]} - {self.time_interval[-1]}')
        plt.savefig(f'{self.path}/bound.png')
        plt.close()
    
    def fixed_band_plot(self, y_true, y_pred):
        # arrange list
        arrange_y_pred = self._arrange_list_band(y_pred)
        arrange_y_pred = [np.array(item).flatten() for item in arrange_y_pred]
        
        arrange_y_true = self._arrange_list_band(y_true)
        arrange_y_true = np.array([item[0] for item in arrange_y_true])
        
        # set parameters
        file_name = f'fixed_{self.time_interval[0]} - {self.time_interval[-1]}'
        palette = sns.color_palette('pastel', len(arrange_y_pred))
        x_val_true = np.arange(arrange_y_true.shape[0])
        upper_bound, lower_bound = self.get_bound(arrange_y_pred)
        
        # plot
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=x_val_true, y=arrange_y_true, color='black')
        for i, item in enumerate(arrange_y_pred):
            x_val_pred = np.repeat(np.array([i]), item.shape[0])
            sns.scatterplot(x=x_val_pred, y=arrange_y_pred[i], alpha=0.2, color=palette[i])
        plt.fill_between(x_val_true, lower_bound, upper_bound, color='gray', alpha=0.8, zorder=10)
            
        plt.xlabel('Time')
        plt.ylabel('Price')
        # plt.xticks(ticks=range(0, y_pred.shape[0], self.seq_len), labels=[self.time_array[0]] * len(self.time_interval))
        # plt.legend().remove()
        plt.savefig(f'{self.path}/bound_fixed.png')
        plt.close()
    
    def single_time_plot(self, y_true, y_pred):
        palette = sns.color_palette('pastel', self.seq_len)
        for i, date in enumerate(self.time_interval):
            if not os.path.exists(f'{self.path}/{date}'): os.mkdir(f'{self.path}/{date}')
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
                upper_bound, lower_bound = self.get_bound(y_hat[j])
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
    
    def fixed_single_time_plot(self, y_true, y_pred):
        """
        plot single time prediction with scatter plot and histogram
        """
        # arrange list
        arrange_y_pred = self._arrange_list_band(y_pred)[(self.seq_len - 1):-self.seq_len]
        
        arrange_y_true = self._arrange_list_band(y_true)
        arrange_y_true = [np.array(item[0]) for item in arrange_y_true]
        
        # set parameters
        palette = sns.color_palette('pastel', self.seq_len)
        for i, date in enumerate(self.time_interval[1:]):
            if not os.path.exists(f'{self.path}/{date}'): os.mkdir(f'{self.path}/{date}')
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
            y = np.array(arrange_y_true[i*self.seq_len:(i + 1)*self.seq_len])
            # band plot
            sns.lineplot(x=x_val, y=y, color='black', ax=ax1)
            for j in range(self.seq_len - 1):
                # process prediction
                y_hat = []
                for idx, sublist in enumerate(arrange_y_pred[j + i * self.seq_len:(i + 1) * self.seq_len]):
                    if idx == 0: y_hat.append(np.array(sublist).flatten())
                    else: y_hat.append(np.array(sublist[:-idx]).flatten())
                
                upper_bound, lower_bound = self.get_bound(y_hat)
                x_val = np.arange(j, self.seq_len)
                x_scatter = np.repeat(x_val, [i.shape[0] for i in y_hat])
                y_scatter = np.array([i for sublist in y_hat for i in sublist])
                sns.scatterplot(x=x_scatter, y=y_scatter, color=palette[j], ax=ax1, alpha=0.3)
                ax1.fill_between(x_val, lower_bound, upper_bound, color=palette[j], alpha=0.7)
                # histogram
                ax2.cla()
                sns.histplot(x=y_hat[0], stat='density', color='blue', alpha=0.3, ax=ax2)
                ax2.axvline(x=y[j], color='black', linestyle='--', linewidth=5)
                
                # # save fig
                ax1.set_title(f'{date} - {self.time_array[j]}')
                plt.savefig(f'{self.path}/{date}/{self.time_array[j]}_fixed.png')
            plt.close()
        
    def validation_chart(self, file_name, y_true, y_pred):
        palette = sns.color_palette('pastel', len(self.time_interval) * self.seq_len)
        plt.figure(figsize=(10, 5))
        for i, (y, y_hat) in enumerate(zip(y_true, y_pred)):
            # set values
            x_val = np.arange(i, i + self.target_len)
            y, y_hat = np.array(y), np.array(y_hat).flatten()
            # ploting
            sns.lineplot(x=x_val, y=y, color='black')
            sns.lineplot(x=x_val, y=np.array(y_hat), color=palette[i])
            
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.savefig(f'{self.path}/{file_name}.png')
        plt.close()
    
    def _get_time_interval(self):
        # 取得時間陣列
        start_time = datetime.strptime('09:01', '%H:%M')
        end_time = datetime.strptime('13:30', '%H:%M')
        x_ticks_interval = self.args.time_step

        time_array = []
        current_time = start_time
        while current_time < end_time:
            time_array.append(current_time.strftime('%H:%M'))
            current_time += timedelta(minutes=x_ticks_interval)
        return time_array
        
    def get_bound(self, y_preds):
        upper_bound = []
        lower_bound = []
        
        for pred in y_preds:
            upper_bound.append(np.percentile(pred, self.args.bound_percent))
            lower_bound.append(np.percentile(pred, 100 - self.args.bound_percent))

        return np.array(upper_bound), np.array(lower_bound)
    
    def _arrange_list_dist(self, array):
        out = [[] for _ in range(self.seq_len)]
        for i, item in enumerate(array):
            idx = i % self.seq_len
            if len(out[idx]) == 0: out[idx] = list(item[0])
            else: out[idx] += list(item[0])
        return np.array(out)
    
    def _arrange_list_dist_fixed(self, list):
        out = []
        for i, item in enumerate(list):
            if i % self.seq_len == 0: out.append(item)
            else:
                head = np.array(item[-(i % self.target_len):])
                tail = np.array(item[:-(i % self.target_len)])
                out.append(np.concatenate((head, tail), axis=0))
        return np.array(out)
    
    def _arrange_list_band(self, list):
        m, n = len(list), len(list[0])
        result = [[] for _ in range(m + n - 1)]

        for i in range(m):
            for j in range(n):
                result[i + j].append(np.array(list[i][j]))
        
        # for item in result:
        #     print(np.array(item).shape)
        return result
