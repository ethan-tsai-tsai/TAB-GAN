import os
import numpy as np
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import rpy2.robjects as ro
from scipy import stats
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import warnings
warnings.filterwarnings('ignore')

from arguments import parse_args
from preprocessor import DataProcessor
from utils import *

class DCCGARCHSimulator:
    def __init__(self, args, data: pd.DataFrame):
        """初始化模擬器"""
        self.args = args
        self.data = data
        self.fit_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.n_series = len(self.fit_columns)
        self.epsilon = 1e-8
        self.prophet_models = {}
        self.decomposition = {}
        
        # 計算價格偏差統計
        self._calculate_price_metrics()
        
        # 初始化 R 環境
        pandas2ri.activate()
        self.rmgarch = importr('rmgarch')
        self.rugarch = importr('rugarch')
        _ = importr('base')
        
        # initialize folder
        self.set_path()

        # transform data
        self.data = self.transform(self.data)
    
    def _calculate_price_metrics(self):
        """計算價格相關的統計指標"""
        # 計算真實平均成交價格
        self.data['true_avg_price'] = self.data['Amount'] / self.data['Volume']
        
        # 計算估計的VWAP
        self.data['estimated_vwap'] = (self.data['Open'] + self.data['High'] + 
                                     self.data['Low'] + self.data['Close']) / 4
        
        # 計算價格偏差率及其統計特性
        price_deviation = ((self.data['true_avg_price'] - self.data['estimated_vwap']) / 
                         self.data['estimated_vwap'])
        
        self.price_stats = {
            'mean_deviation': price_deviation.mean(),
            'std_deviation': price_deviation.std(),
            'percentile_5': price_deviation.quantile(0.05),
            'percentile_95': price_deviation.quantile(0.95)
        }
    
    def set_path(self):
        self.img_path = f'./img/simulate/{self.args.stock}'
        self.data_path = f'./data/simulated'
        if not os.path.exists(self.img_path): os.mkdir(self.img_path)
        else: clear_folder(self.img_path)
        
        if not os.path.exists(self.data_path): os.mkdir(self.data_path)
        
    def transform(self, data):
        self.transform_params = {}
        transformed_data = data.copy()
        self.transform_params = {}
        
        for col in ['Volume']:
            
            col_data = data[col].copy()
            self.transform_params[col] = {
                'min_value': col_data.min(),
                'max_value': col_data.max(),
                'median': col_data.median(),
                'q1': col_data.quantile(0.25),
                'q3': col_data.quantile(0.75)
            }
            
            # 1. 確保數據為正
            col_data = data[col] - data[col].min() + 1
            
            # 2. 對數轉換
            transformed_values = np.log1p(col_data)
            
            # 3. 保存轉換參數
            self.transform_params[col] = {
                'min_value': data[col].min()
            }
            
            transformed_data[col] = transformed_values
            
        return transformed_data

    def inverse_transform(self, data):
        inversed_data = data.copy()
        
        for col in ['Volume']:
            if col in self.transform_params:
                # 1. 獲取參數
                min_value = self.transform_params[col]['min_value']
                
                # 2. 反轉對數轉換
                inversed_values = np.expm1(data[col])
                
                # 3. 還原到原始尺度
                inversed_values = inversed_values + min_value - 1
                
                # 4. 確保不出現負值
                inversed_values = np.maximum(inversed_values, 0)
                
                inversed_data[col] = inversed_values
                
        return inversed_data
        
    def decompose_series(self) -> None:
        """使用 Prophet 對每個序列進行趨勢和季節性分解"""
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            print(f'Fitting column: {col}')
            # 使用轉換後的數據進行分解
            df = pd.DataFrame({
                'ds': self.data.index,
                'y': self.data[col]
            })
            
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                changepoint_prior_scale=0.05
            )
            model.fit(df)
            
            self.prophet_models[col] = model
            
            # 獲取分解結果
            forecast = model.predict(df)
            self.decomposition[col] = {
                'trend': forecast['trend'].values,
                'seasonal': (forecast['yearly'] + forecast['weekly'] + forecast['daily']).values,
                'daily': forecast['daily'].values,
                'residuals': self.data[col].values - forecast['yhat'].values
            }
    
    def fit_dcc_garch(self) -> None:
        """使用 R 的 rmgarch 套件擬合 DCC-GARCH 模型"""
        # 收集所有序列的殘差
        residuals = pd.DataFrame({
            col: self.decomposition[col]['residuals']
            for col in self.fit_columns
        })
        
        # 數據預處理
        print("Checking residuals statistics:")
        print(residuals.describe())
        
        # 標準化殘差
        residuals = (residuals - residuals.mean()) / residuals.std()
        
        # 將 Python 數據轉換為 R 格式
        r_residuals = pandas2ri.py2rpy(residuals)
        
        r_code = """
        function(data, n_series) {
            library(rmgarch)
            
            tryCatch({
                # 檢查數據
                if (any(is.na(data)) || any(is.infinite(as.matrix(data)))) {
                    stop("Data contains NA or Infinite values")
                }
                
                # 1. 先擬合單變量 GARCH
                uspec = multispec(
                    replicate(n_series,
                        ugarchspec(
                            mean.model = list(armaOrder = c(0,0), include.mean = FALSE),
                            variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
                            distribution.model = "sstd",
                            fixed.pars = list()
                        )
                    )
                )
                
                print("Fitting univariate models...")
                fit.uni = multifit(uspec, data, fit.control = list(scale = TRUE))
                
                # 2. build DCC 
                dccspec = dccspec(
                    uspec = uspec, 
                    dccOrder = c(1,1),
                    distribution = "mvt",
                    model = "DCC"
                )
                
                print("Fitting DCC model...")
                # 3. fit DCC
                fit = dccfit(
                    dccspec, 
                    data = data, 
                    fit.control = list(
                        scale = TRUE,
                        eval.se = TRUE
                    ),
                    solver = "solnp",
                    solver.control = list(
                        trace = TRUE,
                        tol = 1e-8,
                        delta = 1e-7,
                        max.major = 1000,
                        max.minor = 1000
                    )
                )
                
                if (is.null(fit)) {
                    stop("DCC fitting returned NULL")
                }
                
                print("DCC fitting completed")
                return(fit)
                
            }, error = function(e) {
                print(paste("Error in fitting:", e$message))
                return(NULL)
            })
        }
        """
        
        # 創建 R 函數並執行
        r_fit_dcc = ro.r(r_code)
        self.dcc_fit = r_fit_dcc(r_residuals, self.n_series)
        
        # 檢查擬合結果
        if self.dcc_fit is None:
            raise Exception("DCC-GARCH model fitting failed")

    def simulate_dcc_garch(self, n_ahead: int) -> np.ndarray:
        """使用擬合的 DCC-GARCH 模型生成模擬數據"""
        if self.dcc_fit is None:
            raise Exception("No fitted DCC-GARCH model available")
            
        r_code = """
        function(fit, n_ahead) {
            tryCatch({
                print("Starting simulation...")
                sim = dccsim(fit, n.sim = n_ahead, n.start = 0)
                if (is.null(sim)) {
                    stop("Simulation returned NULL")
                }
                print("Simulation completed")
                return(fitted(sim))
            }, error = function(e) {
                print(paste("Error in simulation:", e$message))
                return(NULL)
            })
        }
        """
        
        r_simulate = ro.r(r_code)
        sim_result = r_simulate(self.dcc_fit, n_ahead)
        
        if sim_result is None:
            raise Exception("DCC-GARCH simulation failed")
            
        return np.array(sim_result)
    
    def simulate(self, seed: Optional[int] = None) -> pd.DataFrame:
        """生成完整的模擬數據"""
        if seed is not None:
            np.random.seed(seed)
            ro.r('set.seed')(seed)
        
        # 儲存模擬結果
        simulated_data = pd.DataFrame(index=self.data.index, columns=self.fit_columns)
        df_for_prophet = pd.DataFrame({'ds': self.data.index})

        simulated_residuals = self.simulate_dcc_garch(len(self.data.index))
        
        for i, col in enumerate(self.fit_columns):
            # 使用原始數據的日期進行預測
            forecast = self.prophet_models[col].predict(df_for_prophet)
            trend = forecast['trend'].values
            seasonal = (forecast['yearly'] + forecast['weekly'] + forecast['daily']).values
            
            # 將趨勢、季節性和模擬殘差相加
            simulated_data[col] = trend + seasonal + simulated_residuals[:, i]
        
        # 進行異常值處理
        simulated_data = self.clean_outliers(simulated_data, window=5)
    
        # 確保模擬數據的統計特性與原始數據相似
        for col in simulated_data.columns:
            orig_std = self.data[col].std()
            sim_std = simulated_data[col].std()
            
            # 調整波動性
            if sim_std > orig_std * 1.5:  # 如果模擬數據波動過大
                scale_factor = orig_std / sim_std
                simulated_data[col] = simulated_data[col] * scale_factor
        
        # 再次清理異常值
        simulated_data = self.clean_outliers(simulated_data, window=3)
        
        # inverse transform
        simulated_data = self.inverse_transform(simulated_data)
        
        # 根據 Volume 和價格生成 Amount
        simulated_data = self._generate_amount(simulated_data)
        
        simulated_data = self.format_simulated_data(simulated_data)
        return simulated_data
    
    def round_price(self, price):
        """根據台灣股市規則對股價進行四捨五入"""
        if price < 50:
            return round(price, 2)  # 小於50元，最小單位0.01
        elif price < 100:
            return round(price * 20) / 20  # 50-100元，最小單位0.05
        elif price < 500:
            return round(price, 1)  # 100-500元，最小單位0.1
        elif price < 1000:
            return round(price * 2) / 2  # 500-1000元，最小單位0.5
        else:
            return round(price)  # 大於1000元，最小單位1

    def format_simulated_data(self, simulated_data: pd.DataFrame) -> pd.DataFrame:
        """格式化模擬數據以符合真實交易數據格式"""
        formatted_data = simulated_data.copy()
        
        # 處理價格欄位
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            formatted_data[col] = formatted_data[col].apply(self.round_price)
        
        # 驗證並調整價格關係
        formatted_data[price_columns] = formatted_data[price_columns].apply(
            self.validate_price_values, axis=1)
        
        # 處理成交量和金額
        formatted_data['Volume'] = formatted_data['Volume'].round().astype(int)
        formatted_data['Amount'] = formatted_data['Amount'].round().astype(int)
        
        # 確保成交量和金額為正數
        formatted_data['Volume'] = formatted_data['Volume'].clip(lower=0)
        formatted_data['Amount'] = formatted_data['Amount'].clip(lower=0)
        
        return formatted_data
    
    def validate_price_values(self, row):
        """驗證價格高低關係是否合理"""
        high = row['High']
        low = row['Low']
        open_price = row['Open']
        close = row['Close']
        
        # 調整不合理的價格關係
        if high < max(open_price, close):
            high = max(open_price, close)
        if low > min(open_price, close):
            low = min(open_price, close)
        
        return pd.Series({
            'High': high,
            'Low': low,
            'Open': open_price,
            'Close': close
        })
    
    def _generate_amount(self, simulated_data: pd.DataFrame) -> pd.DataFrame:
        """根據成交量和價格生成成交金額"""
        result = simulated_data.copy()
        
        # 1. 計算估計的VWAP
        estimated_vwap = (result['Open'] + result['High'] + 
                         result['Low'] + result['Close']) / 4
        
        # 2. 生成受限的隨機價格偏差
        n_samples = len(result)
        price_deviations = np.random.normal(
            self.price_stats['mean_deviation'],
            self.price_stats['std_deviation'],
            size=n_samples
        )
        
        # 限制在合理範圍內
        price_deviations = np.clip(
            price_deviations,
            self.price_stats['percentile_5'],
            self.price_stats['percentile_95']
        )
        
        # 3. 計算模擬的平均交易價格
        simulated_avg_price = estimated_vwap * (1 + price_deviations)
        
        # 4. 生成成交金額
        result['Amount'] = result['Volume'] * simulated_avg_price
        
        return result
    
    def validate_correlation(self, simulated_data: pd.DataFrame) -> None:
        """驗證原始數據和模擬數據的相關性結構"""
        
        original_corr = self.data[['Open', 'High', 'Low', 'Close', 'Volume', 'Amount']].corr()
        simulated_corr = simulated_data.corr()

        # 繪製熱圖比較
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        im1 = ax1.imshow(original_corr)
        ax1.set_title('Original Correlation')
        plt.colorbar(im1, ax=ax1)
        
        im2 = ax2.imshow(simulated_corr)
        ax2.set_title('Simulated Correlation')
        plt.colorbar(im2, ax=ax2)
        
        # 設置標籤
        col_list = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount']

        for ax in [ax1, ax2]:
            ax.set_xticks(range(len(col_list)))
            ax.set_yticks(range(len(col_list)))
            ax.set_xticklabels(col_list, rotation=45)
            ax.set_yticklabels(col_list)
        
        plt.tight_layout()
        plt.savefig(f'./img/simulate/{args.stock}/corr.png')
        plt.close()
        # 印出相關係數的差異
        diff = original_corr - simulated_corr
        print("\nCorrelation Difference (Original - Simulated):")
        print(diff)
    
    def validate_statistics(self, simulated_data: pd.DataFrame) -> None:
        """
        驗證模擬數據的統計特性
        """
        for col in simulated_data.columns:
            print(f"\nValidating {col}:")
            print("\nOriginal Data Statistics:")
            print(self.data[col].describe())
            print("\nSimulated Data Statistics:")
            print(simulated_data[col].describe())
            
            # 繪製原始數據和模擬數據的分布對比圖
            plt.figure(figsize=(15, 6))
            
            # 左邊繪製原始尺度的箱線圖
            plt.subplot(121)
            plt.boxplot([self.data[col], simulated_data[col]], labels=['Original', 'Simulated'])
            plt.title(f'{col} Distribution (Original Scale)')
            plt.yscale('log')  # 使用對數刻度更容易觀察
            plt.grid(True)
            
            # 右邊繪製對數尺度的密度圖
            plt.subplot(122)
            sns.kdeplot(data=np.log(self.data[col] + self.epsilon), label='Original')
            sns.kdeplot(data=np.log(simulated_data[col] + self.epsilon), label='Simulated')
            plt.title(f'{col} Distribution (Log Scale)')
            plt.xlabel(f'Log({col})')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.img_path, f'distribution_{col}.png'))
            plt.close()
    
    def validate_volume_amount(self, simulated_data: pd.DataFrame) -> None:
        """驗證Volume和Amount之間的關係"""
        # 計算原始數據和模擬數據的平均價格
        orig_avg_price = self.data['Amount'] / self.data['Volume']
        sim_avg_price = simulated_data['Amount'] / simulated_data['Volume']
        
        print("\nPrice Statistics:")
        print("Original Data:")
        print(orig_avg_price.describe())
        print("\nSimulated Data:")
        print(sim_avg_price.describe())
        
        # 計算相關係數
        orig_corr = np.corrcoef(self.data['Volume'], self.data['Amount'])[0,1]
        sim_corr = np.corrcoef(simulated_data['Volume'], simulated_data['Amount'])[0,1]
        
        print("\nVolume-Amount Correlation:")
        print(f"Original: {orig_corr:.4f}")
        print(f"Simulated: {sim_corr:.4f}")
        
        # 繪製散點圖比較
        plt.figure(figsize=(12, 5))
        
        plt.subplot(121)
        plt.scatter(np.log(self.data['Volume']), np.log(self.data['Amount']), 
                   alpha=0.5, label='Original')
        plt.title('Original Data')
        plt.xlabel('Log(Volume)')
        plt.ylabel('Log(Amount)')
        plt.grid(True)
        
        plt.subplot(122)
        plt.scatter(np.log(simulated_data['Volume']), 
                   np.log(simulated_data['Amount']), 
                   alpha=0.5, label='Simulated')
        plt.title('Simulated Data')
        plt.xlabel('Log(Volume)')
        plt.ylabel('Log(Amount)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.img_path, 'volume_amount_relationship.png'))
        plt.close()
    
    def clean_outliers(self, simulated_data: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        使用移動中位數平滑異常波動
        """
        cleaned_data = simulated_data.copy()
        
        for col in ['Volume']:
            # 計算移動中位數
            rolling_median = cleaned_data[col].rolling(window=window, center=True).median()
            rolling_std = cleaned_data[col].rolling(window=window, center=True).std()
            
            # 識別異常值 (使用對數尺度計算)
            # log_values = np.log(cleaned_data[col] + self.epsilon)
            # log_median = np.log(rolling_median + self.epsilon)
            # log_std = np.log(rolling_std + self.epsilon)
            
            z_scores = np.abs((cleaned_data[col] - rolling_median) / rolling_std)
            outliers = z_scores > 3.0  # 超過3個標準差視為異常
            
            # 使用移動中位數替換異常值
            cleaned_data.loc[outliers, col] = rolling_median[outliers]
        
        return cleaned_data
    
    def plot_simulation(self, simulated_data: pd.DataFrame) -> None:
        """繪製原始數據和模擬數據的比較圖"""
        x_col = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount']
        n_cols = len(x_col)
        fig, axes = plt.subplots(n_cols, 1, figsize=(12, 4*n_cols))
        
        if n_cols == 1:
            axes = [axes]
            
        for i, col in enumerate(x_col):
            # 繪製原始數據
            axes[i].plot(self.data.index, self.data[col], 
                        label='Historical', color='blue')
            
            # 繪製模擬數據
            axes[i].plot(simulated_data.index, simulated_data[col], 
                        label='Simulated', color='red', linestyle='--')
            
            axes[i].set_title(f'{col} - Historical vs Simulated')
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.img_path, f'simulation.png'))
        plt.close()

    def plot_decomposition(self, col: str = None) -> None:
        """視覺化分解結果，包括日內模式"""
        if col is not None:
            columns = [col]
        else:
            columns = self.fit_columns
            
        for col in columns:
            # 創建包含5個子圖的圖表（添加日內模式）
            _, axes = plt.subplots(5, 1, figsize=(15, 25))
            
            # 繪製原始數據
            axes[0].plot(self.data.index, self.data[col], label='Original Data', color='blue')
            axes[0].set_title(f'{col} - Original Time Series')
            axes[0].grid(True)
            axes[0].legend()
            
            # 繪製趨勢
            axes[1].plot(self.data.index, self.decomposition[col]['trend'], 
                        label='Trend', color='red')
            axes[1].set_title(f'{col} - Trend Component')
            axes[1].grid(True)
            axes[1].legend()
            
            # 繪製季節性（年度+週度）
            axes[2].plot(self.data.index, self.decomposition[col]['seasonal'], 
                        label='Seasonal', color='green')
            axes[2].set_title(f'{col} - Combined Seasonal Component')
            axes[2].grid(True)
            axes[2].legend()
            
            # 繪製日內模式
            axes[3].plot(self.data.index, self.decomposition[col]['daily'],
                        label='Daily Pattern', color='orange')
            axes[3].set_title(f'{col} - Daily Seasonality')
            axes[3].grid(True)
            axes[3].legend()
            
            # 繪製殘差
            axes[4].plot(self.data.index, self.decomposition[col]['residuals'], 
                        label='Residuals', color='purple')
            axes[4].set_title(f'{col} - Residuals')
            axes[4].grid(True)
            axes[4].legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.img_path, f'decomposition_{col}.png'))
            plt.close()

    def plot_seasonal_components(self, col: str = None) -> None:
        """視覺化季節性成分"""
        if col is not None:
            columns = [col]
        else:
            columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                
        for col in columns:
            model = self.prophet_models[col]
            df = pd.DataFrame({'ds': self.data.index})
            forecast = model.predict(df)
            
            # 創建包含3個子圖的圖表
            _, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
            
            # 繪製週季節性
            week_seasonal = forecast['weekly'].values
            week_days = pd.date_range(start='2024-01-01', periods=len(week_seasonal), freq='D')
            ax1.plot(week_days[:7], week_seasonal[:7], color='blue', marker='o')
            ax1.set_title(f'{col} - Weekly Seasonality')
            ax1.set_xlabel('Day of Week')
            ax1.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
            ax1.grid(True)
            
            # 繪製年季節性
            year_seasonal = forecast['yearly'].values
            year_days = pd.date_range(start='2024-01-01', periods=len(year_seasonal), freq='D')
            ax2.plot(year_days[:365], year_seasonal[:365], color='green')
            ax2.set_title(f'{col} - Yearly Seasonality')
            ax2.set_xlabel('Month')
            ax2.set_xticks([year_days[0] + pd.DateOffset(months=i) for i in range(12)])
            ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            ax2.grid(True)
            
            # 繪製日內模式
            daily_seasonal = forecast['daily'].values
            # hours = np.linspace(0, 9, len(daily_seasonal))
            ax3.plot(range(9), daily_seasonal[:9], color='orange', marker='o')
            ax3.set_title(f'{col} - Daily Seasonality')
            ax3.set_xlabel('Hour of Day')
            ax3.set_xticks(range(1, 10))
            ax3.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.img_path, f'seasonality_{col}.png'))
            plt.close()

    def plot_all_components(self) -> None:
        """繪製所有變數的分解圖和季節性圖"""
        
        # 繪製所有分解圖
        self.plot_decomposition()
        
        # 繪製所有季節性圖
        self.plot_seasonal_components()

if __name__ == '__main__':
    # 初始化模擬器
    args = parse_args()
    args.mode = 'simulate'
    processor = DataProcessor(args)
    data = processor.get_data()

    simulator = DCCGARCHSimulator(args, data)

    # 進行分解和擬合
    simulator.decompose_series()
    simulator.plot_all_components()
    simulator.fit_dcc_garch()

    # 生成模擬數據
    simulated_data = simulator.simulate(seed=42)
    simulated_data.to_csv(f'./data/simulated/{args.stock}_simulated.csv')
    # 驗證結果
    simulator.validate_correlation(simulated_data)
    simulator.validate_statistics(simulated_data)
    simulator.validate_volume_amount(simulated_data)

    simulator.plot_simulation(simulated_data)