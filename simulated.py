import numpy as np
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import warnings
warnings.filterwarnings('ignore')

from arguments import parse_args
from preprocessor import DataProcessor
from utils import *

class DCCGARCHSimulator:
    def __init__(self, data: pd.DataFrame):
        """
        初始化模擬器
        
        Parameters:
        -----------
        data : pd.DataFrame
            輸入數據，index 為時間戳，columns 為不同資產
        """
        self.data = data
        self.n_series = len(data.columns)
        self.prophet_models = {}
        self.decomposition = {}
        
        # 初始化 R 環境
        pandas2ri.activate()
        self.rmgarch = importr('rmgarch')
        self.rugarch = importr('rugarch')
        base = importr('base')
        
    def decompose_series(self) -> None:
        """使用 Prophet 對每個序列進行趨勢和季節性分解"""
        for col in self.data.columns:
            print(f'Fitting column: {col}')
            # 準備 Prophet 數據
            df = pd.DataFrame({
                'ds': self.data.index,
                'y': self.data[col]
            })
            
            # 擬合 Prophet 模型
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            model.fit(df)
            
            self.prophet_models[col] = model
            
            # 獲取分解結果
            forecast = model.predict(df)
            self.decomposition[col] = {
                'trend': forecast['trend'].values,
                'seasonal': (forecast['yearly'] + forecast['weekly']).values,
                'residuals': self.data[col].values - forecast['yhat'].values
            }
    
    def fit_dcc_garch(self) -> None:
        """使用 R 的 rmgarch 套件擬合 DCC-GARCH 模型"""
        # 收集所有序列的殘差
        residuals = pd.DataFrame({
            col: self.decomposition[col]['residuals']
            for col in self.data.columns
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
                            distribution.model = "norm",
                            fixed.pars = list()
                        )
                    )
                )
                
                print("Fitting univariate models...")
                fit.uni = multifit(uspec, data, fit.control = list(scale = TRUE))
                
                # 2. 創建 DCC 規格
                dccspec = dccspec(
                    uspec = uspec, 
                    dccOrder = c(1,1),
                    distribution = "mvnorm",
                    model = "DCC"
                )
                
                print("Fitting DCC model...")
                # 3. 使用更穩定的設定擬合 DCC
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
        
    def simulate_dcc_garch(self, n_ahead: int) -> np.ndarray:
        """
        使用擬合的 DCC-GARCH 模型生成模擬數據
        
        Parameters:
        -----------
        n_ahead : int
            預測期數
            
        Returns:
        --------
        np.ndarray
            模擬的殘差
        """
        # R 代碼：執行模擬
        r_code = """
        function(fit, n_ahead) {
            sim = dccsim(fit, n.sim = n_ahead)
            return(fitted(sim))
        }
        """
        
        r_simulate = ro.r(r_code)
        simulated_residuals = np.array(r_simulate(self.dcc_fit, n_ahead))
        
        return simulated_residuals
    
    def simulate(self, n_ahead: int, seed: Optional[int] = None) -> pd.DataFrame:
        """
        生成完整的模擬數據
        
        Parameters:
        -----------
        n_ahead : int
            預測期數
        seed : int, optional
            隨機種子
            
        Returns:
        --------
        pd.DataFrame
            模擬的時間序列數據
        """
        if seed is not None:
            np.random.seed(seed)
            ro.r('set.seed')(seed)
            
        # 生成未來日期
        future_dates = pd.date_range(
            start=self.data.index[-1] + timedelta(days=1),
            periods=n_ahead,
            freq='B'  # 營業日
        )
        
        # 儲存模擬結果
        simulated_data = pd.DataFrame(index=future_dates, columns=self.data.columns)
        future_df = pd.DataFrame({'ds': future_dates})
        
        # 使用 DCC-GARCH 模擬殘差
        simulated_residuals = self.simulate_dcc_garch(n_ahead)
        
        # 對每個序列進行預測
        for i, col in enumerate(self.data.columns):
            # Prophet 預測趨勢和季節性
            forecast = self.prophet_models[col].predict(future_df)
            trend = forecast['trend'].values
            seasonal = (forecast['yearly'] + forecast['weekly']).values
            
            # 組合所有組件
            simulated_data[col] = trend + seasonal + simulated_residuals[:, i]
        
        return simulated_data
    
    def validate_correlation(self, simulated_data: pd.DataFrame) -> None:
        """驗證原始數據和模擬數據的相關性結構"""
        # 計算原始數據的相關係數
        original_corr = self.data.corr()
        
        # 計算模擬數據的相關係數
        simulated_corr = simulated_data.corr()
        
        # 繪製熱圖比較
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        im1 = ax1.imshow(original_corr)
        ax1.set_title('Original Correlation')
        plt.colorbar(im1, ax=ax1)
        
        im2 = ax2.imshow(simulated_corr)
        ax2.set_title('Simulated Correlation')
        plt.colorbar(im2, ax=ax2)
        
        # 設置標籤
        for ax in [ax1, ax2]:
            ax.set_xticks(range(len(self.data.columns)))
            ax.set_yticks(range(len(self.data.columns)))
            ax.set_xticklabels(self.data.columns, rotation=45)
            ax.set_yticklabels(self.data.columns)
        
        plt.tight_layout()
        plt.savefig(f'./img/simulate/{args.stock}/corr.png')
        
        # 印出相關係數的差異
        diff = original_corr - simulated_corr
        print("\nCorrelation Difference (Original - Simulated):")
        print(diff)
    
    def plot_simulation(self, simulated_data: pd.DataFrame) -> None:
        """
        繪製原始數據和模擬數據的比較圖
        
        Parameters:
        -----------
        simulated_data : pd.DataFrame
            模擬的數據
        """
        n_cols = len(self.data.columns)
        fig, axes = plt.subplots(n_cols, 1, figsize=(12, 4*n_cols))
        
        if n_cols == 1:
            axes = [axes]
            
        for i, col in enumerate(self.data.columns):
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
        plt.savefig(f'./img/simulate/{args.stock}/compare.png')

    def plot_decomposition(self, col: str = None) -> None:
        """
        視覺化 Prophet 分解的趨勢和季節性成分
        
        Parameters:
        -----------
        col : str, optional
            指定要繪製的列名，若為 None 則繪製所有列
        """
        if col is not None:
            columns = [col]
        else:
            columns = self.data.columns
            
        for col in columns:
            # 創建包含4個子圖的圖表
            fig, axes = plt.subplots(4, 1, figsize=(15, 20))
            
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
            
            # 繪製季節性
            axes[2].plot(self.data.index, self.decomposition[col]['seasonal'], 
                        label='Seasonal', color='green')
            axes[2].set_title(f'{col} - Seasonal Component')
            axes[2].grid(True)
            axes[2].legend()
            
            # 繪製殘差
            axes[3].plot(self.data.index, self.decomposition[col]['residuals'], 
                        label='Residuals', color='purple')
            axes[3].set_title(f'{col} - Residuals')
            axes[3].grid(True)
            axes[3].legend()
            
            # 調整布局
            plt.tight_layout()
            
            # 儲存圖片
            plt.savefig(f'./img/simulate/{args.stock}/decomposition_{col}.png')
            plt.close()

    def plot_seasonal_components(self, col: str = None) -> None:
        """
        單獨視覺化週季節性和年季節性成分
        
        Parameters:
        -----------
        col : str, optional
            指定要繪製的列名，若為 None 則繪製所有列
        """
        if col is not None:
            columns = [col]
        else:
            columns = self.data.columns
            
        for col in columns:
            # 獲取 Prophet 模型
            model = self.prophet_models[col]
            
            # 生成完整預測，包含季節性成分
            df = pd.DataFrame({'ds': self.data.index})
            forecast = model.predict(df)
            
            # 創建包含2個子圖的圖表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
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
            
            # 調整布局
            plt.tight_layout()
            
            # 儲存圖片
            plt.savefig(f'./img/simulate/{args.stock}/seasonality_{col}.png')
            plt.close()

    def plot_all_components(self) -> None:
        """
        繪製所有變數的分解圖和季節性圖
        """
        # 檢查圖片保存目錄是否存在
        import os
        os.makedirs(f'./img/simulate/{args.stock}', exist_ok=True)
        
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

    simulator = DCCGARCHSimulator(data)

    # 進行分解和擬合
    simulator.decompose_series()
    simulator.plot_all_components()
    simulator.fit_dcc_garch()

    # 生成模擬數據
    n_ahead = 365  # 模擬一年的交易日數據
    simulated_data = simulator.simulate(n_ahead, seed=42)
    simulated_data.to_csv(f'./data/simulated/{args.stock}_simulated.csv')
    # 驗證結果
    simulator.validate_correlation(simulated_data)
    simulator.plot_simulation(simulated_data)