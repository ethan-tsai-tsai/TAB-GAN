import numpy as np
from scipy.special import rel_entr

def calc_kld(generated_data, ground_truth, bins=50, epsilon=1e-10):
    y_true = np.array(ground_truth)
    y_pred = np.array(generated_data)

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))

    hist_true, edges = np.histogram(y_true, bins=bins, range=(min_val, max_val), density=True)
    hist_pred, _ = np.histogram(y_pred, bins=edges, density=True)

    hist_true = hist_true + epsilon
    hist_pred = hist_pred + epsilon

    hist_true = hist_true / np.sum(hist_true)
    hist_pred = hist_pred / np.sum(hist_pred)

    kld = np.sum(rel_entr(hist_true, hist_pred))
    
    return kld

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

