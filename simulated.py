import os
import numpy as np
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from arguments import parse_args
from preprocessor import DataProcessor

class StockSimulator:
    def __init__(self, args, trading_days=0):
        """
        初始化股票模擬器
        
        Parameters:
        trading_days: 要模擬的交易天數
        """
        self.args = args
        self.trading_days = trading_days
        self.setup_paths()
        self.setup_r_environment()
        self.load_data()
        
    def setup_paths(self):
        """設置相關路徑"""
        self.data_path = f'./data/{self.args.stock}_data.csv'
        self.save_path = f'./data/simulated/{self.args.stock}'
        self.img_path = f'./img/simulate/{self.args.stock}'
        
        # 創建必要的目錄
        for path in [self.save_path, self.img_path]:
            if not os.path.exists(path):
                os.makedirs(path)
    
    def setup_r_environment(self):
        """設置 R 環境"""
        pandas2ri.activate()
        self.rmgarch = importr('rmgarch')
    
    def load_data(self):
        """載入原始數據"""
        processor = DataProcessor(args)
        self.df = processor.get_data()
        self.df['ts'] = self.df.index
        self.cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount']
        
        # 計算各欄位的統計特性
        self.calc_data_statistics()
    
    def calc_data_statistics(self):
        """計算原始數據的統計特性"""
        self.stats = {}
        
        for col in self.cols:
            self.stats[col] = {
                'mean': self.df[col].mean(),
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                'max': self.df[col].max(),
                'q01': self.df[col].quantile(0.01),
                'q99': self.df[col].quantile(0.99),
                'autocorr': self.df[col].autocorr()
            }
            
            # 計算對數收益率的統計特性
            if col != 'Volume' and col != 'Amount':
                returns = np.log(self.df[col]).diff()
                self.stats[f'{col}_returns'] = {
                    'mean': returns.mean(),
                    'std': returns.std(),
                    'autocorr': returns.autocorr()
                }
        
        # 計算價格之間的相對關係
        self.calc_price_relations()
    
    def calc_price_relations(self):
        """計算價格之間的關係統計"""
        # 計算High/Low相對於中間價格的偏離程度
        mid_price = (self.df['Open'] + self.df['Close']) / 2
        self.stats['price_relations'] = {
            'high_deviation': ((self.df['High'] - mid_price) / mid_price).mean(),
            'low_deviation': ((mid_price - self.df['Low']) / mid_price).mean(),
            'volume_price_ratio': (self.df['Volume'] * self.df['Close'] / self.df['Amount']).median()
        }
    
    def apply_volatility_constraints(self, simulated):
        """應用波動率約束"""
        # 使用移動窗口計算歷史波動率
        window = 20
        hist_vol = self.df['Close'].pct_change().rolling(window).std()
        avg_vol = hist_vol.mean()
        max_vol = hist_vol.quantile(0.99)
        
        # 限制模擬數據的波動率
        returns = pd.Series(simulated).pct_change()
        rolling_vol = returns.rolling(window).std()
        
        # 如果波動率超過限制，進行調整
        scale_factor = np.minimum(1, max_vol / rolling_vol)
        adjusted_returns = returns * scale_factor
        
        # 轉換回價格序列
        adjusted_price = np.exp(np.log(simulated[0]) + np.cumsum(adjusted_returns))
        
        return adjusted_price
        
    def create_trading_future_df(self, model):
        """生成交易日的未來時間序列"""
        periods = self.trading_days * 9
        future = model.make_future_dataframe(periods=periods, freq='30min')
        
        future['time_of_day'] = future['ds'].dt.time
        future['weekday'] = future['ds'].dt.weekday
        
        start_time = pd.to_datetime('09:01').time()
        end_time = pd.to_datetime('13:01').time()
        
        mask = (
            (future['time_of_day'] >= start_time) & 
            (future['time_of_day'] <= end_time)
        )
        filtered_future = future[mask].copy()
        filtered_future = filtered_future.reset_index(drop=True)[['ds']]
        
        return filtered_future
    
    def generate_garch_errors(self, data, n_sim):
        """使用改進的 GARCH 模型生成誤差項"""
        r_data = pandas2ri.py2rpy(pd.DataFrame(data))
        
        r_code = """
        function(data, n_sim) {
            library(rmgarch)
            
            # 使用更穩定的 GARCH 模型設定
            spec <- ugarchspec(
                variance.model = list(
                    model = "sGARCH",
                    garchOrder = c(1,1)
                ),
                mean.model = list(
                    armaOrder = c(0,0),
                    include.mean = TRUE
                ),
                distribution.model = "std"  # 使用 t 分布
            )
            
            # 使用更多的最佳化選項
            ctrl = list(
                tol = 1e-6,           # 容忍度
                delta = 1e-7,         # delta
                maxiter = 500,        # 最大迭代次數
                scale = 1,            # 縮放因子
                rec.init = 'all'      # 遞迴初始化
            )
            
            tryCatch({
                fit <- ugarchfit(spec, data[,1], 
                               solver = 'hybrid',
                               solver.control = ctrl,
                               fit.control = list(scale = 1))
                
                sim <- ugarchsim(fit, n.sim = n_sim)
                vol <- sigma(sim)
                
                # 確保返回合理的值
                vol[is.na(vol)] <- mean(vol, na.rm = TRUE)
                vol[vol <= 0] <- mean(vol[vol > 0])
                
                return(as.numeric(vol))
            }, error = function(e) {
                # 如果模型擬合失敗，使用簡單的移動標準差
                e
                sd <- sd(data[,1], na.rm = TRUE)
                return(rep(sd, n_sim))
            })
        }
        """
        
        try:
            r_func = robjects.r(r_code)
            errors = np.array(r_func(r_data, n_sim))
            
            # 確保誤差項是合理的
            if np.any(np.isnan(errors)):
                print("警告: 發現 NaN 值，使用均值替代")
                errors = np.nan_to_num(errors, nan=np.nanmean(errors))
            
            # 標準化誤差項
            errors = errors / np.std(errors) * np.std(data)
            
            return errors.flatten()
            
        except Exception as e:
            print(f"GARCH 模型執行錯誤: {str(e)}")
            return np.random.normal(0, np.std(data), n_sim)
    
    def plot_components(self, col, forecast, errors):
        """繪製並保存組件分解圖"""
        plt.style.use('default')
        
        # 創建子圖
        fig, axes = plt.subplots(4, 1, figsize=(15, 20))
        fig.suptitle(f'{self.args.stock} {col}', fontsize=16, y=0.95)
        
        # 1. 趨勢圖
        axes[0].plot(forecast['ds'], forecast['trend'], 'b-', linewidth=2)
        axes[0].set_title('Trend')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Value')
        axes[0].grid(True)
        
        # 2. 年度季節性
        axes[1].plot(forecast['ds'], forecast['yearly'], 'g-', linewidth=2)
        axes[1].set_title('Yearly seasonal')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Value')
        axes[1].grid(True)
        
        # 3. 日內季節性
        axes[2].plot(forecast['ds'], forecast['daily'], 'r-', linewidth=2)
        axes[2].set_title('Daily seasonal')
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('Value')
        axes[2].grid(True)
        
        # 4. 誤差項
        axes[3].plot(forecast['ds'], errors, 'k-', linewidth=2)
        axes[3].set_title('GARCH error')
        axes[3].set_xlabel('Date')
        axes[3].set_ylabel('value')
        axes[3].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.img_path}/{col}_components.png')
        plt.close()
    
    def simulate_price_series(self, col):
        """模擬單一價格序列"""
        print(f"模擬 {col} 序列...")
        
        # 準備數據
        prophet_data = self.df[['ts', col]].copy()
        prophet_data.columns = ['ds', 'y']
        
        # 使用對數轉換處理價格數據
        if col not in ['Volume', 'Amount']:
            prophet_data['y'] = np.log(prophet_data['y'])
        
        # Prophet模型設置
        model = Prophet(
            interval_width=0.95,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05  # 降低以減少過擬合
        )
        model.fit(prophet_data)
        
        # 生成未來時間序列
        future = self.create_trading_future_df(model)
        forecast = model.predict(future)
        
        # 生成考慮自相關的誤差項
        errors = self.generate_correlated_errors(
            prophet_data['y'], 
            len(future), 
            self.stats[col]['autocorr'] if col in ['Volume', 'Amount'] else self.stats[f'{col}_returns']['autocorr']
        )
        
        # 合成序列
        simulated = (forecast['trend'].values + 
                    forecast['yearly'].values + 
                    forecast['weekly'].values + 
                    forecast['daily'].values + 
                    errors)
        
        # 如果是價格數據，需要轉換回原始尺度
        if col not in ['Volume', 'Amount']:
            simulated = np.exp(simulated)
            # 應用波動率約束
            simulated = self.apply_volatility_constraints(simulated)
        else:
            # 對Volume和Amount使用正態化處理
            simulated = stats.norm.cdf(simulated)  # 轉換到[0,1]區間
            # 映射到原始數據的範圍
            simulated = (self.stats[col]['q99'] - self.stats[col]['q01']) * simulated + self.stats[col]['q01']
        
        self.plot_components(col, forecast, errors)
        
        return simulated
    
    def generate_correlated_errors(self, data, n_sim, autocorr):
        """生成具有自相關性的誤差項"""
        # 使用AR(1)過程生成具有自相關的誤差
        errors = np.zeros(n_sim)
        errors[0] = np.random.normal(0, np.std(data))
        
        for i in range(1, n_sim):
            errors[i] = autocorr * errors[i-1] + np.random.normal(0, np.std(data) * np.sqrt(1 - autocorr**2))
        
        return errors
    
    def simulate_all(self):
        """模擬所有價格序列"""
        raw_simulated = pd.DataFrame()
        raw_simulated['ts'] = self.df['ts']

        for col in self.cols:
            raw_simulated[col] = self.simulate_price_series(col)
        
        # 調整數據以符合邏輯約束
        adjusted_data = self.adjust_price_series(raw_simulated)
        
        # 驗證數據合理性
        self.validate_data(adjusted_data)
        
        # 保存模擬數據
        adjusted_data.to_csv(f'{self.save_path}/simulated_prices.csv', index=False)
        print("模擬完成！")
        
        return adjusted_data
    
    def plot_comparison(self, simulated_data):
        """繪製原始數據和模擬數據的比較圖"""
        plt.style.use('default')
        
        for col in self.cols:
            plt.figure(figsize=(15, 8))
            plt.plot(self.df['ts'], self.df[col], 'b-', label='raw data', alpha=0.6)
            plt.plot(simulated_data['ts'], simulated_data[col], 'r-', label='simulated', alpha=0.6)
            plt.title(f'{self.args.stock} {col} original v.s. simulated data')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{self.img_path}/{col}_comparison.png')
            plt.close()
            
    def validate_data(self, data):
        """驗證數據是否符合邏輯約束和統計特性"""
        issues = []
        
        # 檢查基本邏輯約束
        if not all(data['High'] >= data['Open']):
            issues.append("發現High小於Open的情況")
        if not all(data['High'] >= data['Close']):
            issues.append("發現High小於Close的情況")
        if not all(data['Low'] <= data['Open']):
            issues.append("發現Low大於Open的情況")
        if not all(data['Low'] <= data['Close']):
            issues.append("發現Low大於Close的情況")
        
        # 檢查數值範圍
        for col in self.cols:
            if data[col].min() < self.stats[col]['q01']:
                issues.append(f"{col}出現異常低值")
            if data[col].max() > self.stats[col]['q99']:
                issues.append(f"{col}出現異常高值")
        
        # 檢查自相關性
        for col in self.cols:
            sim_autocorr = data[col].autocorr()
            orig_autocorr = self.stats[col]['autocorr']
            if abs(sim_autocorr - orig_autocorr) > 0.3:
                issues.append(f"{col}的自相關性與原始數據差異過大")
        
        # 報告問題
        if issues:
            print("警告: 發現以下數據問題:")
            for issue in issues:
                print(f"- {issue}")

    def adjust_price_series(self, simulated_data):
        """調整價格序列以符合邏輯約束"""
        adjusted = pd.DataFrame()
        adjusted['ts'] = simulated_data['ts']
        
        # 使用移動平均平滑Close價格
        window = 5
        adjusted['Close'] = pd.Series(simulated_data['Close']).rolling(window=window, min_periods=1).mean()
        
        # 根據歷史關係生成其他價格
        for t in range(len(adjusted)):
            # 生成符合自相關的價格範圍
            high_dev = self.stats['price_relations']['high_deviation']
            low_dev = self.stats['price_relations']['low_deviation']
            
            if t > 0:
                # 考慮前一時刻的價格
                prev_high = adjusted['High'].iloc[t-1]
                prev_low = adjusted['Low'].iloc[t-1]
                current_close = adjusted['Close'].iloc[t]
                
                # 使用加權平均生成新的high/low
                weight = 0.7  # 控制連續性的權重
                high_range = weight * prev_high + (1-weight) * (current_close * (1 + high_dev))
                low_range = weight * prev_low + (1-weight) * (current_close * (1 - low_dev))
            else:
                # 第一個時刻
                current_close = adjusted['Close'].iloc[t]
                high_range = current_close * (1 + high_dev)
                low_range = current_close * (1 - low_dev)
            
            # 確保價格邏輯
            adjusted.loc[t, 'High'] = max(high_range, adjusted['Close'].iloc[t])
            adjusted.loc[t, 'Low'] = min(low_range, adjusted['Close'].iloc[t])
            
            # 生成Open價格
            if t > 0:
                # 考慮前一個Close價格
                prev_close = adjusted['Close'].iloc[t-1]
                open_weight = 0.3
                adjusted.loc[t, 'Open'] = (open_weight * prev_close + 
                                         (1-open_weight) * (adjusted['Low'].iloc[t] + 
                                         np.random.uniform(0, 1) * (adjusted['High'].iloc[t] - adjusted['Low'].iloc[t])))
            else:
                adjusted.loc[t, 'Open'] = adjusted['Low'].iloc[t] + np.random.uniform(0, 1) * (adjusted['High'].iloc[t] - adjusted['Low'].iloc[t])
        
        # 處理Volume和Amount
        adjusted['Volume'] = simulated_data['Volume']
        
        # 根據價格調整Amount
        mid_price = (adjusted['Open'] + adjusted['Close']) / 2
        adjusted['Amount'] = adjusted['Volume'] * mid_price * self.stats['price_relations']['volume_price_ratio']
        
        return adjusted

if __name__ == '__main__':
    # 設置參數
    args = parse_args()
    args.mode = 'simulate'
    
    # 創建模擬器實例
    simulator = StockSimulator(args)
    
    # 執行模擬
    simulated_data = simulator.simulate_all()
    
    # 繪製比較圖
    simulator.plot_comparison(simulated_data)