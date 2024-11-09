import os
import numpy as np
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
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
        
        # 建立和擬合 Prophet 模型
        model = Prophet(
            interval_width=0.95,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode='multiplicative'
        )
        model.fit(prophet_data)
        
        # 生成未來時間序列
        future = self.create_trading_future_df(model)
        
        # 預測趨勢和季節性
        forecast = model.predict(future)
        
        # 生成誤差項
        errors = self.generate_garch_errors(prophet_data['y'], len(future))
        
        # 繪製組件分解圖
        self.plot_components(col, forecast, errors)
        
        # 合成最終序列
        simulated = (forecast['trend'].values + 
                    forecast['yearly'].values + 
                    forecast['weekly'].values + 
                    forecast['daily'].values + 
                    errors.flatten())
        
        return simulated
    
    def simulate_all(self):
        """模擬所有價格序列"""
        simulated_data = pd.DataFrame()
        simulated_data['ts'] = self.df['ts']

        for col in self.cols:
            simulated_data[col] = self.simulate_price_series(col)
        
        # 保存模擬數據
        simulated_data.to_csv(f'{self.save_path}/simulated_prices.csv', index=False)
        print("模擬完成！")
        
        return simulated_data
    
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