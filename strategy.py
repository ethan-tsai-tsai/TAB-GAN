import os
import torch
import pickle
import numpy as np
import pandas as pd
from typing import Tuple

from arguments import parse_args
from preprocessor import StockDataset, DataProcessor
from utils import plot_predicions, TechnicalIndicators, TradingStrategy, visualize_band
from train import wgan

class TradingAnalysis:
    def __init__(self, args, trial):
        self.args = args
        self.args.name = f'trial_{trial}'
        self.bound_percent = [50, 70, 90]
        self.strategy = TradingStrategy()
        self.FILE_NAME = f'{self.args.stock}_{self.args.name}'
        self.device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
        
    def load_model_args(self) -> None:
        """載入模型參數"""
        checkpoint = torch.load(f'./model/{self.FILE_NAME}/final.pth')
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
            pred_upper += self.get_values_between(upper[0], upper[1])
            pred_lower += self.get_values_between(lower[0], lower[1])
            
        return pred_upper, pred_lower
    
    def caculate_bolling_band(self, upper, lower):
        upper_bound = []
        lower_bound = []
        for i in range(len(upper) - 1):
            upper_bound += self.get_values_between(upper[i], upper[i+1])
            lower_bound += self.get_values_between(lower[i], lower[i+1])

        return upper_bound, lower_bound
    
    def get_values_between(self, start, end):
        # 計算步長
        step = (end - start) / (self.args.time_step - 1)
        values = [start + step * i for i in range(self.args.time_step)]
        
        return values
    
    def bound_results(self):
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
        strategy_data = pd.read_csv(f'./data/{self.args.stock}/strategy.csv')
        
        with open(f'./data/{self.args.stock}/scaler_y.pkl', 'rb') as f:
            scaler_y = pickle.load(f)
        test_data['y'] = scaler_y.inverse_transform(test_data[['y']])
        
        window_size = 270 // self.args.time_step * self.args.window_size
        
        # 布林通道策略
        upper, lower = TechnicalIndicators.bollinger_band(
            pd.concat([test_data['y'], pd.Series(strategy_data.iloc[-1, 4])]), 
            window_size
        )
        upper, lower = list(upper[window_size:]), list(lower[window_size:])
        booling_upper, booling_lower = self.caculate_bolling_band(upper, lower)
        
        # 準備交易數據
        trading = strategy_data.iloc[:, [0, 4]].copy()
        trading['bolling_upper'] = booling_upper
        trading['bolling_lower'] = booling_lower
        
        # 預測界限策略
        for bound_percent in self.bound_percent:
            plot_util.args.bound_percent = bound_percent
            pred_upper, pred_lower = self.calculate_bounds(y_preds, plot_util)
            trading[f'pred_upper_{bound_percent}'] = pred_upper
            trading[f'pred_lower_{bound_percent}'] = pred_lower
        
        return trading
    
    def analyze_trading_strategies(self, trading) -> pd.DataFrame:
        # 生成訊號
        for bound_percent in self.bound_percent:
            trading[f'pred_signals_{bound_percent}'] = self.strategy.generate_signals(
                np.array(trading['Close']), 
                np.array(trading[f'pred_upper_{bound_percent}']), 
                np.array(trading[f'pred_lower_{bound_percent}'])
            )
        trading['bolling_signals'] = self.strategy.generate_signals(
            np.array(trading['Close']), 
            np.array(trading['bolling_upper']), 
            np.array(trading['bolling_lower'])
        )
        # 儲存所有策略的績效指標
        all_metrics = []
        strategy_names = []
        
        # 計算布林通道策略的績效
        
        bolling_metrics = self.strategy.calculate_returns(
            np.array(trading['Close']), 
            np.array(trading['bolling_signals'])
        )
        all_metrics.append(bolling_metrics)
        strategy_names.append('Bollinger Bands')
        
        # 計算各個 bound_percent 的預測通道績效
        for bound_percent in self.bound_percent:
            pred_metrics = self.strategy.calculate_returns(
                np.array(trading['Close']), 
                np.array(trading[f'pred_signals_{bound_percent}'])
            )
            all_metrics.append(pred_metrics)
            strategy_names.append(f'Prediction Bands {bound_percent}%')
        
        # 整理所有結果到 DataFrame
        results = pd.DataFrame({
            'Strategy': strategy_names,
            'Total Return': [metrics[0] for metrics in all_metrics],
            'Annual Return': [metrics[1] for metrics in all_metrics],
            'Sharpe Ratio': [metrics[2] for metrics in all_metrics]
        })
        
        return results

if __name__ == '__main__':
    args = parse_args()
    args.mode = 'trial'
    
    # 執行分析
    print(f'Starting trial with stock {args.stock}')
    num_trials = 12
    trading_data = pd.DataFrame()
    for trial in range(1, num_trials + 1):
        print(f'Running trial {trial}......')
        analyzer = TradingAnalysis(args, trial)
        _ = DataProcessor(args, trial)
        trading = analyzer.bound_results()
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
    results.to_csv(f'./data/trading_results_{args.stock}.csv', index=False)
    trading_data.to_csv(f'./data/trading_signals_{args.stock}.csv', index=False)
    
    visualize_band(args)