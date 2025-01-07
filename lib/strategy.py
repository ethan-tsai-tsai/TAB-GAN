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
        
        # 手續費率和交易稅率設定
        self.buy_fee_rate = 0.001425 * 0.1  # 買入手續費率 0.1425% * 手續費優惠 0.1
        self.sell_fee_rate = 0.001425 * 0.1  # 賣出手續費率 0.1425% * 手續費優惠 0.1
        self.normal_tax_rate = 0.003  # 一般交易稅率 0.3%
        self.day_trade_tax_rate = 0.0015  # 當沖交易稅率 0.15%
        
        # 止損點設定
        self.stop_loss = -0.1  # 止損點
        
    def calculate_sell_cost(self, is_day_trade: bool) -> float:
        """
        計算賣出成本（手續費 + 交易稅）
        """
        tax_rate = self.day_trade_tax_rate if is_day_trade else self.normal_tax_rate
        return self.sell_fee_rate + tax_rate
        
    def generate_signals(self, actual_prices: np.ndarray, upper_bounds: np.ndarray, 
                        lower_bounds: np.ndarray) -> np.ndarray:
        """
        根據實際價格和上下界生成交易訊號，結合成本、止損和上界三個條件
        
        Args:
            actual_prices (np.ndarray): 實際價格序列
            upper_bounds (np.ndarray): 上界序列
            lower_bounds (np.ndarray): 下界序列
            
        Returns:
            np.ndarray: 交易訊號序列 (1: 買入, -1: 賣出, 0: 不動作)
        """
        signals = []
        current_position = 0
        last_trade_time = -np.inf
        minutes_per_day = 270
        
        for t in range(len(actual_prices)):
            signal = 0
            
            if current_position == 0:
                if actual_prices[t] < lower_bounds[t] * 0.98:
                    signal = 1
                    current_position = 1
                    last_trade_time = t
                    
            else:
                # 計算實際獲利
                is_day_trade = (t - last_trade_time) < minutes_per_day
                sell_cost = self.calculate_sell_cost(is_day_trade)
                actual_profit = (actual_prices[t] - actual_prices[last_trade_time]) / actual_prices[last_trade_time]
                
                # 賣出條件
                if ((actual_prices[t] > upper_bounds[t] and actual_profit > sell_cost) or 
                    (actual_profit < self.stop_loss)):
                    signal = -1
                    current_position = 0
                    last_trade_time = t
            
            signals.append(signal)
        
        return np.array(signals)
    
    def calculate_returns(self, prices: np.ndarray, signals: np.ndarray) -> Tuple[float, float, float]:
        """
        計算交易績效指標，使用日級數據，包含手續費和交易稅
        當沖交易使用較低的交易稅率(0.15%)
        """
        # 將分鐘數據轉換為日級數據
        daily_returns = []
        last_buy_price = None
        last_buy_time = None  # 記錄最後買入的時間
        daily_positions = []  # 記錄每日的交易
        minutes_per_day = 270  # 每日交易分鐘數

        for i in range(len(prices)):
            if signals[i] == 1:
                last_buy_price = prices[i] * (1 + self.buy_fee_rate)  # 實際買入成本
                last_buy_time = i  # 記錄買入時間
            elif signals[i] == -1 and last_buy_price is not None:
                # 判斷是否為當沖交易
                is_day_trade = (i // minutes_per_day) == (last_buy_time // minutes_per_day)
                tax_rate = self.day_trade_tax_rate if is_day_trade else self.normal_tax_rate
                
                sell_price = prices[i] * (1 - self.sell_fee_rate - tax_rate)
                returns = (sell_price - last_buy_price) / last_buy_price
                day_index = i // minutes_per_day
                while day_index >= len(daily_positions):
                    daily_positions.append([])
                daily_positions[day_index].append(returns)
                last_buy_price = None
                last_buy_time = None

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