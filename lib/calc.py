import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

def calc_kld(generated_data, ground_truth, bins=50, epsilon=1e-5):
    # 計算範圍
    ground_truth = np.array(ground_truth)
    generated_data = np.array(generated_data)
    
    all_data = np.concatenate([ground_truth, generated_data])
    range_min = np.percentile(all_data, 1)
    range_max = np.percentile(all_data, 99)
    
    # 計算直方圖
    pd_gt, _ = np.histogram(ground_truth, bins=bins, density=True, range=(range_min, range_max))
    pd_gen, _ = np.histogram(generated_data, bins=bins, density=True, range=(range_min, range_max))
    
    # 添加平滑值避免除零問題
    pd_gt = pd_gt + epsilon
    pd_gen = pd_gen + epsilon
    
    # 重新正規化確保概率和為1
    pd_gt = pd_gt / np.sum(pd_gt)
    pd_gen = pd_gen / np.sum(pd_gen)
    
    # 計算 KLD
    kld = 0
    for x1, x2 in zip(pd_gt, pd_gen):
        kld += x1 * np.log(x1 / x2)
    
    return kld

class KLDLoss(nn.Module):
    """
    計算序列資料的 Kullback-Leibler Divergence Loss
    輸入形狀: [batch_size, seq_len]
    """
    def __init__(self, reduction='mean', eps=1e-8):
        """
        Args:
            reduction (str): 'none', 'mean', 或 'sum'
            eps (float): 避免除以零的小數值
        """
        super(KLDLoss, self).__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target):
        """
        計算 KLD loss
        
        Args:
            pred (torch.Tensor): 預測機率分布, shape [batch_size, seq_len]
            target (torch.Tensor): 目標機率分布, shape [batch_size, seq_len]
            
        Returns:
            torch.Tensor: KLD loss
        """
        # 確保輸入為機率分布 (和為1)
        pred = F.softmax(pred, dim=-1)
        target = F.softmax(target, dim=-1)
        
        # 避免log(0)，加入eps
        pred = torch.clamp(pred, min=self.eps)
        target = torch.clamp(target, min=self.eps)
        
        # 計算 KLD: target * log(target/pred)
        kld = target * torch.log(target/pred)
        
        # 對序列維度求和
        kld = torch.sum(kld, dim=-1)
        
        # 根據reduction方式處理batch維度
        if self.reduction == 'none':
            return kld
        elif self.reduction == 'mean':
            return torch.mean(kld)
        else:  # sum
            return torch.sum(kld)

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

