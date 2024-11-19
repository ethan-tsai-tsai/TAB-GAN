import os
import torch
import pickle
import numpy as np
import pandas as pd
from typing import Tuple

from arguments import parse_args
from preprocessor import StockDataset, DataProcessor
from utils import plot_predicions, TechnicalIndicators, TradingStrategy
from train import wgan
from trial import verify_data

class TradingAnalysis:
    def __init__(self, args, trial):
        """
        初始化交易分析類
        
        Args:
            args: 參數配置
        """
        self.args = args
        self.args.name = f'trial_{trial}'
        self.strategy = TradingStrategy()
        self.FILE_NAME = f'{self.args.stock}_{self.args.name}'
        self.device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
        
    def load_model_args(self) -> None:
        """載入模型參數"""
        checkpoint = torch.load(f'./model/{self.FILE_NAME}/final.pth')
        
        # 嘗試從參數檔案載入
        if os.path.exists(f'./model/{self.FILE_NAME}_args.pkl'):
            with open(f'./model/{self.FILE_NAME}_args.pkl', 'rb') as f:
                saved_args = pickle.load(f)
            for key, value in saved_args.items():
                if hasattr(self.args, key):
                    setattr(self.args, key, value)
        else:
            # 從checkpoint載入
            for key, value in checkpoint['args'].items(): 
                if key not in ['pred_times', 'bound_percent']:
                    setattr(self.args, key, value)
                    
        return checkpoint
        
    def prepare_data(self) -> Tuple[torch.Tensor, np.ndarray, StockDataset]:
        """準備數據"""
        test_datasets = StockDataset(self.args, f'./data/{self.args.stock}/test.csv')
        
        X = torch.tensor(np.array(test_datasets.X), dtype=torch.float32)
        y = np.array(test_datasets.y)
        y[y == -10] = np.nan
        
        return X, y, test_datasets
    
    def calculate_bounds(self, y_preds: np.ndarray, plot_util: plot_predicions) -> Tuple[list, list]:
        """計算預測的上下界"""
        pred_upper, pred_lower = [], []
        for pred in y_preds:
            upper, lower = plot_util.get_bound(pred)
            pred_upper.append(upper[0])
            pred_lower.append(lower[0])
        return pred_upper, pred_lower
    
    def trading_results(self):
        """分析交易策略"""
        # 載入模型和參數
        checkpoint = self.load_model_args()
        
        # 準備數據
        X, y, test_datasets = self.prepare_data()
        
        # 設置模型和預測
        time_interval = test_datasets.time_intervals[self.args.window_size:]
        wgan_model = wgan(test_datasets, self.args)
        wgan_model.model_g.load_state_dict(checkpoint['model_g'])
        
        # 預測
        plot_util = plot_predicions(
            path=f'./img/{self.FILE_NAME}', 
            args=self.args, 
            time_interval=time_interval
        )
        y_preds, _ = wgan_model.predict(X, y)
        
        # 交易策略分析
        test_data = pd.read_csv(f'./data/{self.args.stock}/test.csv')
        window_size = 270 // self.args.time_step * self.args.window_size
        
        # 布林通道策略
        upper, lower = TechnicalIndicators.bollinger_band(
            test_data['Close'], 
            window_size
        )
        
        # 準備交易數據
        trading = test_data.iloc[window_size:, [0, 4]].copy()  # 只選取必要的列
        trading['bolling_upper'] = upper[window_size:]
        trading['bolling_lower'] = lower[window_size:]
        
        # 預測界限策略
        pred_upper, pred_lower = self.calculate_bounds(y_preds, plot_util)
        trading['pred_upper'] = pred_upper
        trading['pred_lower'] = pred_lower
        
        # 生成訊號
        trading['bolling_signals'] = self.strategy.generate_signals(
            np.array(trading['Close']), 
            np.array(trading['bolling_upper']), 
            np.array(trading['bolling_lower'])
        )
        
        trading['pred_signals'] = self.strategy.generate_signals(
            np.array(trading['Close']), 
            np.array(trading['pred_upper']), 
            np.array(trading['pred_lower'])
        )
        
        return trading
    
    def analyze_trading_strategies(self, trading) -> pd.DataFrame:
        
        # 計算績效
        bolling_metrics = self.strategy.calculate_returns(
            np.array(trading['Close']), 
            np.array(trading['bolling_signals'])
        )
        
        pred_metrics = self.strategy.calculate_returns(
            np.array(trading['Close']), 
            np.array(trading['pred_signals'])
        )
        
        # 整理結果
        results = pd.DataFrame({
            'Strategy': ['Bollinger Bands', 'Prediction Bands'],
            'Total Return': [bolling_metrics[0], pred_metrics[0]],
            'Annual Return': [bolling_metrics[1], pred_metrics[1]],
            'Sharpe Ratio': [bolling_metrics[2], pred_metrics[2]]
        })
        
        return results

if __name__ == '__main__':
    args = parse_args()
    args.mode = 'trial'
    
    # 執行分析
    num_trials = 1
    trading_data = pd.DataFrame()
    for trial in range(1, num_trials + 1):
        analyzer = TradingAnalysis(args, trial)
        print(analyzer.FILE_NAME)
        _ = DataProcessor(args, trial)
        verify_data(args, trial)
        trading = analyzer.trading_results()
        if trial == 0:
            trading_data = trading
        else:
            trading_data = pd.concat([trading, trading_data], axis=0)
    
    results = analyzer.analyze_trading_strategies(trading_data)
    
    # 顯示結果
    print("\nTrading Strategy Results:")
    print("------------------------")
    print(results.to_string(index=False))
    
    # 儲存結果
    results.to_csv(f'trading_results_{args.stock}.csv', index=False)
    trading_data.to_csv(f'trading_signals_{args.stock}.csv', index=False)