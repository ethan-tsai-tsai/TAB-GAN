import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from typing import Tuple, List, Dict
from datetime import datetime, timedelta
from scipy.linalg import sqrtm
from scipy.special import rel_entr

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # 計算均值差的平方
    diff = mu1 - mu2
    diff_squared = np.sum(diff ** 2)

    # 計算協方差矩陣的乘積的平方根
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)

    # 確保協方差矩陣的數值穩定性
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # 計算 FID
    fid_value = diff_squared + np.trace(sigma1 + sigma2 - 2 * covmean)

    return fid_value

def calculate_statistics(data):
    mu = np.mean(data, axis=0)
    sigma = np.cov(data, rowvar=False)
    return mu, sigma

def fid_score(generated_data, real_data):
    # 計算真實數據和生成數據的均值和協方差
    mu1, sigma1 = calculate_statistics(real_data)
    mu2, sigma2 = calculate_statistics(generated_data)

    # 計算 FID
    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_value

def calc_kld(generated_data, ground_truth, bins=50, epsilon=1e-10):
    # Find the range that covers both datasets
    min_val = min(np.min(generated_data), np.min(ground_truth))
    max_val = max(np.max(generated_data), np.max(ground_truth))
    
    # Create consistent bins for both histograms
    bins = np.linspace(min_val, max_val, bins+1)
    
    # Calculate histograms with the same bins
    pd_gt, _ = np.histogram(ground_truth, bins=bins, density=True)
    pd_gen, _ = np.histogram(generated_data, bins=bins, density=True)
    
    # Add small constant to avoid division by zero
    pd_gt = pd_gt + epsilon
    pd_gen = pd_gen + epsilon
    
    # Normalize to ensure proper probability distributions
    pd_gt = pd_gt / np.sum(pd_gt)
    pd_gen = pd_gen / np.sum(pd_gen)
    
    # Calculate KL divergence using scipy's rel_entr
    # rel_entr computes x * log(x/y) element-wise
    kld = np.sum(rel_entr(pd_gt, pd_gen))
    
    return kld

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
    plt.savefig(f'./logs/{args.stock}_{args.name}/loss.png')
    
def save_model(model_d, model_g, args, file_name):
    # filtering args
    filter_val = ['noise_dim', 
                  'epoch', 'batch_size', 
                  'hidden_dim_g', 'num_layers_g', 'lr_g',
                  'hidden_dim_d', 'num_layers_d', 'lr_d',
                  'd_iter', 'gp_lambda']
    args = {key:value for key, value in vars(args).items() if key in filter_val}
    torch.save({
        'args': args,
        'model_d': model_d.state_dict(),
        'model_g': model_g.state_dict()
    }, file_name)

def clear_folder(folder_path):
    # Make sure the folder exists
    if not os.path.exists(folder_path):
        print(f"資料夾 {folder_path} 不存在。")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                clear_folder(file_path)
                os.rmdir(file_path)
        except Exception as e:
            print(f"when delete {file_path} has error: {e}")

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

class TechnicalIndicators:
    @staticmethod
    def SMA(data, windows):
        """Simple Moving Average"""
        return data.rolling(window=windows).mean()

    @staticmethod
    def EMA(data, windows):
        """Exponential Moving Average"""
        return data.ewm(span=windows).mean()

    @staticmethod
    def MACD(data, long, short, windows):
        """Moving Average Convergence Divergence"""
        short_ = data.ewm(span=short).mean()
        long_ = data.ewm(span=long).mean()
        macd_ = short_ - long_
        return macd_.ewm(span=windows).mean()

    @staticmethod
    def RSI(data, windows):
        """Relative Strength Index"""
        delta = data.diff(1)
        up = delta.copy()
        down = delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        avg_up = up.rolling(window=windows).mean()
        avg_down = down.rolling(window=windows).mean()
        rs = avg_up / avg_down
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def ATR(high, low, windows):
        """Average True Range"""
        range_ = high - low
        return range_.rolling(window=windows).mean()

    @staticmethod
    def bollinger_band(data, windows):
        """Bollinger Bands"""
        sma = data.rolling(window=windows).mean()
        std = data.rolling(window=windows).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        return upper, lower

    @staticmethod
    def RSV(data, windows):
        """Raw Stochastic Value"""
        min_ = data.rolling(window=windows).min()
        max_ = data.rolling(window=windows).max()
        return (data - min_) / (max_ - min_) * 100

    @staticmethod
    def add_all_indicators(df):
        """Add all technical indicators to the dataframe"""
        data = df.copy()
        
        # Moving Averages
        for window in [7, 14, 21]:
            data[f'{window}ma'] = TechnicalIndicators.EMA(data['Close'], window)
        
        # MACD
        data['7macd'] = TechnicalIndicators.MACD(data['Close'], 3, 11, 7)
        data['14macd'] = TechnicalIndicators.MACD(data['Close'], 7, 21, 14)
        
        # ATR
        for window in [7, 14, 21]:
            data[f'{window}atr'] = TechnicalIndicators.ATR(data['High'], data['Low'], window)
        
        # Bollinger Bands
        for window in [7, 14, 21]:
            upper, lower = TechnicalIndicators.bollinger_band(data['Close'], window)
            data[f'{window}upper'] = upper
            data[f'{window}lower'] = lower
        
        return data

class TradingStrategy:
    def __init__(self, risk_free_rate: float = 0.02):
        """
        初始化交易策略
        
        Args:
            risk_free_rate (float): 無風險利率，用於計算夏普率
        """
        self.risk_free_rate = risk_free_rate
        self.returns = []
        
    def generate_signals(self, actual_prices: np.ndarray, upper_bounds: np.ndarray, 
                        lower_bounds: np.ndarray) -> np.ndarray:
        """
        根據實際價格和上下界生成交易訊號
        
        Args:
            actual_prices (np.ndarray): 實際價格序列
            upper_bounds (np.ndarray): 上界序列
            lower_bounds (np.ndarray): 下界序列
            
        Returns:
            np.ndarray: 交易訊號序列 (1: 買入, -1: 賣出, 0: 不動作)
        """
        signals = []
        current_position = 0
        
        for t in range(len(actual_prices)):
            if actual_prices[t] < lower_bounds[t] and current_position == 0:
                signals.append(1)  # 買入訊號
                current_position = 1
            elif actual_prices[t] > upper_bounds[t] and current_position == 1:
                signals.append(-1)  # 賣出訊號
                current_position = 0
            else:
                signals.append(0)  # 不動作
        
        return np.array(signals)
    
    def calculate_returns(self, prices: np.ndarray, signals: np.ndarray) -> Tuple[float, float, float]:
        """
        計算交易績效指標
        
        Args:
            prices (np.ndarray): 價格序列
            
        Returns:
            Tuple[float, float, float]: (總報酬率, 年化報酬率, 夏普率)
        """
        # 計算每次交易的報酬率
        daily_returns = []
        last_buy_price = None
        
        for i in range(len(prices)):
            if signals[i] == 1:  # 買入
                last_buy_price = prices[i]
            elif signals[i] == -1 and last_buy_price is not None:  # 賣出
                returns = (prices[i] - last_buy_price) / last_buy_price
                daily_returns.append(returns)
                last_buy_price = None
        
        self.returns = daily_returns
        
        # 如果沒有交易，返回零值
        if not daily_returns:
            return 0.0, 0.0, 0.0
        
        # 計算總報酬率
        total_return = (1 + np.array(daily_returns)).prod() - 1
        
        # 計算年化報酬率 (假設252個交易日)
        n_days = len(prices)
        annual_return = (1 + total_return) ** (252 / n_days) - 1
        
        # 計算夏普率
        returns_std = np.std(daily_returns) * np.sqrt(252)  # 年化標準差
        if returns_std == 0:
            sharpe_ratio = 0
        else:
            sharpe_ratio = (annual_return - self.risk_free_rate) / returns_std
        
        return total_return, annual_return, sharpe_ratio