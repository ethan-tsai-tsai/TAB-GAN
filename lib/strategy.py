# import packages
import numpy as np
from typing import Tuple

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
        計算交易績效指標，使用日級數據
        """
        # 將分鐘數據轉換為日級數據
        daily_returns = []
        last_buy_price = None
        daily_positions = []  # 記錄每日的交易
        minutes_per_day = 270  # 每日交易分鐘數

        for i in range(len(prices)):
            if signals[i] == 1:
                last_buy_price = prices[i]
            elif signals[i] == -1 and last_buy_price is not None:
                returns = (prices[i] - last_buy_price) / last_buy_price
                day_index = i // minutes_per_day
                while day_index >= len(daily_positions):
                    daily_positions.append([])
                daily_positions[day_index].append(returns)
                last_buy_price = None

        # 計算每日平均報酬率
        for day_trades in daily_positions:
            if day_trades:
                daily_returns.append(sum(day_trades))

        self.returns = daily_returns

        if not daily_returns:
            return 0.0, 0.0, 0.0

        # 計算總報酬率
        total_return = (1 + np.array(daily_returns)).prod() - 1

        # 計算年化報酬率
        trading_days = len(daily_positions)
        years = trading_days / 252  # 轉換為年
        annual_return = (1 + total_return) ** (1 / years) - 1

        # 計算年化標準差
        returns_std = np.std(daily_returns) * np.sqrt(252)  # 使用252個交易日年化

        # 計算夏普指標
        sharpe_ratio = (annual_return - self.risk_free_rate) / returns_std if returns_std != 0 else 0

        return total_return, annual_return, sharpe_ratio
    