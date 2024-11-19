import os
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from datetime import datetime
from utils import *
from arguments import *

class DataProcessor:
    def __init__(self, args, trial=1):
        self.args = args
        self.trial = trial
        if self.args.mode not in ['trial']:
            print('Processing data ......')
        start_time = datetime.now()
        
        # 創建資料夾
        path = f'./data/{args.stock}'
        if not os.path.exists(path): os.makedirs(path)
        
        # 初始化數據
        self._initialize_data()
        
        # 根據模式處理數據
        if args.mode == 'simulate':
            self._process_simulation_data()
        else:
            self._process_training_data()
        
        if self.args.mode not in ['trial']:
            print(f'Data processing spent {(datetime.now() - start_time).total_seconds(): 2f} seconds')

    def _clear_previous_data(self):
        """清理前一個 trial 的資料和變數"""
        # 清理所有實例變數
        for attr in list(self.__dict__.keys()):
            if attr not in ['args', 'trial']:
                delattr(self, attr)
    
    def _initialize_data(self):
        """初始化基本數據處理"""
        
        # 清理任何可能的類別變數
        self._clear_previous_data()
        
        # load data
        self.data = pd.read_csv(f'./data/{self.args.stock}.csv').sort_values(by='ts')
        
        # 取得參數
        self.time_step = self.args.time_step
        self.target_length = self.args.target_length // self.args.time_step
        self.seq_len = self.args.window_size * (270 // self.args.time_step)
        self.window_size = self.args.window_size
        self.window_stride = self.args.window_stride
        
        # 基本資料處理
        self.data['ts'] = pd.to_datetime(self.data['ts'])
        self.data.set_index('ts', inplace=True)
        self.data = self.data.groupby(self.data.index).mean()
        
        # 補全缺失值
        self.time_intervals = self.data.index.strftime('%Y-%m-%d').unique().tolist()
        self._complete_data()
        self.data = self.data.iloc[::self.time_step, :]

    def _process_simulation_data(self):
        """處理模擬模式的數據"""
        return self.data
    
    def _process_training_data(self):
        """處理訓練模式的數據"""
        
        # 加入技術指標
        self._add_technical_indicators()
        
        # 後續處理
        self.data = self.data.iloc[(270//self.args.time_step)*30::, :]
        self._add_additional_features()
        self._split_and_save_data()
    
    def _add_technical_indicators(self):
        """添加技術指標"""
        self.data = TechnicalIndicators.add_all_indicators(self.data)
    
    def _add_additional_features(self):
        """添加額外特徵"""
        self.data['y'] = self.data['Close']
        self.data['change'] = self._price_change(self.data['y'])
        
        # Date Columns
        self.data['year'] = self.data.index.to_series().dt.year
        self.data['month'] = self.data.index.to_series().dt.month
        self.data['day'] = self.data.index.to_series().dt.day
        self.data['week'] = self.data.index.to_series().dt.weekday
        self.data['hour'] = self.data.index.to_series().dt.hour
        self.data['minute'] = self.data.index.to_series().dt.minute
        
        # 重排列欄位
        columns = [col for col in self.data.columns if col != 'y'] + ['y']
        self.data = self.data[columns]
    
    def _split_and_save_data(self):
        """分割並保存數據"""
        path = f'./data/{self.args.stock}'
        seq_len = 270//self.args.time_step
        window_size = 5
        test_idx = (window_size * self.trial) * seq_len
        if self.trial == 1: test_dataframe = self.data.iloc[-(test_idx + seq_len * self.args.window_size):, :].copy()
        else: test_dataframe = self.data.iloc[-(test_idx + seq_len * self.args.window_size):-(test_idx - seq_len * window_size), :].copy()
        train_dataframe = self.data.iloc[-(test_idx + seq_len * 365 * 2):-test_idx, :].copy()
        
        # Standardize
        col_list = list(self.data.columns.drop('y'))
        scaler_X = MinMaxScaler(feature_range=(-1, 1))
        scaler_y = MinMaxScaler(feature_range=(-1, 1))
        train_dataframe[col_list] = scaler_X.fit_transform(train_dataframe[col_list].values)
        train_dataframe['y'] = scaler_y.fit_transform(train_dataframe[['y']].values)
        test_dataframe[col_list] = scaler_X.fit_transform(test_dataframe[col_list].values)
        test_dataframe['y'] = scaler_y.fit_transform(test_dataframe[['y']].values)
        
        # save scaler and data
        with open(f'{path}/scaler_X.pkl', 'wb') as f:
            pickle.dump(scaler_X, f)
        with open(f'{path}/scaler_y.pkl', 'wb') as f:
            pickle.dump(scaler_y, f)
        train_dataframe.to_csv(f'{path}/train.csv')
        test_dataframe.to_csv(f'{path}/test.csv')
 
    def _price_change(self, data):
        changes = [0]
        for i in range(1, len(data)):
            if data.iloc[i - 1] == 0:
                changes.append(0)
            else:
                change = (data.iloc[i] - data.iloc[i - 1]) / data.iloc[i - 1] * 100
                changes.append(change)
        return changes   
 
    def _complete_data(self):
        # reset index
        new_idx = []
        for date in self.time_intervals:
            start_time = pd.to_datetime(date + ' 09:01:00')
            end_time = pd.to_datetime(date +' 13:30:00')
            datetime_range = pd.date_range(start_time, end_time, freq='min')
            new_idx += list(datetime_range)
        self.data = self.data.reindex(new_idx)
        
        # fill missing values
        self.data['Open'] = self.data['Open'].ffill().interpolate(method='linear')
        self.data['High'] = self.data['High'].ffill().interpolate(method='linear')
        self.data['Low'] = self.data['Low'].ffill().interpolate(method='linear')
        self.data['Close'] = self.data['Close'].ffill().interpolate(method='linear')
        self.data['Volume'] = self.data['Volume'].ffill().interpolate(method='linear')
        self.data['Amount'] = self.data['Amount'].ffill().interpolate(method='linear')

    def get_data(self):
        """獲取處理後的數據"""
        return self.data
    
class StockDataset(Dataset):
    def __init__(self, args, csv_file):
        # setting parameters
        self._args = args
        self.time_step = args.time_step
        self.target_length = args.target_length // args.time_step
        self.seq_len = args.window_size * (270 // args.time_step)
        self.window_size = args.window_size
        self.window_stride = args.window_stride
        
        # read data and set index
        self.data = pd.read_csv(csv_file)
        self.data['ts'] = pd.to_datetime(self.data['ts'])
        self.data.set_index('ts', inplace=True)
        
        # get data information
        self.num_features = len(self.data.columns)
        self.time_intervals = self.data.index.strftime('%Y-%m-%d').unique().tolist()

        self._rolling_window()
    
    def _rolling_window(self):
        self.X, self.y = [], []
        for i in range(0, len(self.data) - self.seq_len, self.window_stride):
            self.X.append(self.data.iloc[i:i+self.seq_len+1, :len(self.data.columns)-1].values)
            if 'y' in self.data.columns:
                y_val = self.data.iloc[i+self.seq_len:i+self.seq_len+self.target_length, len(self.data.columns)-1].values
                if len(y_val) < self.target_length:
                    padding_length = self.target_length - len(y_val)
                    y_val = np.concatenate((y_val, np.full(padding_length, -10)))
                self.y.append(y_val)
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

if __name__ == '__main__':
    args = parse_args()
    data_processor = DataProcessor(args, trial=1)