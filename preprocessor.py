import os
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime
from utils import *
from arguments import *

class DataProcessor:
    def __init__(self, args):
        path = f'./data/{args.stock}'
        if not os.path.exists(path): os.makedirs(path)
        # else: clear_folder(path)
               
        self.args = args
        
        print('Processing data ......')
        start_time = datetime.now()
        
        # load data
        self.data = pd.read_csv(f'./data/{args.stock}.csv').sort_values(by='ts')
        
        # 取得參數
        self.time_step = args.time_step # 每隔幾分鐘取出一筆資料
        self.target_length = args.target_length // args.time_step # y的資料筆數 
        self.seq_len = args.window_size * (270 // args.time_step) # X的資料筆數
        self.window_size = args.window_size # 移動窗格大小
        self.window_stride = args.window_stride # 移動窗格的移動步伐
        
        # 資料處理
        self.data['ts'] = pd.to_datetime(self.data['ts'])
        self.data.set_index(self.data['ts'], inplace=True)
        self.data = self.data.groupby(self.data.index).mean() # 重複日期
        
         # 補全缺失值
        self.time_intervals = self.data.index.strftime('%Y-%m-%d').unique().tolist() # 資料中的日期
        self._complete_data() 
        self.data = self.data[self.data.index > '2021-01-01']
        
         # 加入欄位
        # Technical Indicators
        # self.data['cmf'] = self._cmf()
        # self.data['colose_ratio'] = self._close_ratio(window=self.time_step)
        # self.data['volume_percentile'] = self._volume_percentile(window=self.time_step)
        
        # 切割掉第一天（技術指標大多沒有值）
        self.data = self.data.iloc[270::, :]
        
        # ※根據處理資料的方式，分為在切割資料前和後
        self.data = self.data.iloc[::self.time_step, :] # 每 time_step 分鐘取一筆資料
        
        self.data['y'] = self.data['Close']
        self.data['change'] = self._price_change(self.data['y'])
        
        ## Date Columns
        self.data['year'] = self.data.index.to_series().dt.year
        self.data['month'] = self.data.index.to_series().dt.month
        self.data['day'] = self.data.index.to_series().dt.day
        self.data['week'] = self.data.index.to_series().dt.weekday
        self.data['hour'] = self.data.index.to_series().dt.hour
        self.data['minute'] = self.data.index.to_series().dt.minute
        
        columns = [col for col in self.data.columns if col != 'y'] + ['y'] # arrange columns
        self.data = self.data[columns]
        
        # prevent error
        assert not self.data.isnull().values.any(), 'There are missing values in the data.'
        assert not np.isinf(self.data.values).any(), 'There are inf values in the data.'

        # split and save dataframe
        test_dataframe = self.data.iloc[-(270//self.args.time_step) * 8:, :]
        train_dataframe = self.data.iloc[:-(270//self.args.time_step) * 5, :]
        val_dataframe = self.data[(self.data.index.month==8) & (self.data.index.year==2024)]

        train_dataframe.to_csv(f'{path}/train.csv')
        test_dataframe.to_csv(f'{path}/test.csv')
        val_dataframe.to_csv(f'{path}/val.csv')

        end_time = datetime.now()
        print(f'Data processing spent {(end_time - start_time).total_seconds(): 2f} seconds')

    def _money_flow_multiplier(self):
        rolling_high = self.data['High'].rolling(window=self.time_step).max().fillna(0)
        rolling_low = self.data['Low'].rolling(window=self.time_step).min().fillna(0)
        money_flow_multiplier = ((self.data['Close'] - rolling_low) - (rolling_high - self.data['Close'])) / (rolling_high - rolling_low)
        return money_flow_multiplier
        
    def _money_flow_volume(self):
        money_flow_multiplier = self._money_flow_multiplier()
        rolling_volume = self.data['Volume'].rolling(window=self.time_step).sum().fillna(0)
        money_flow_volume = money_flow_multiplier * rolling_volume
        return money_flow_volume
        
    def _cmf(self):
        money_flow_volume = self._money_flow_volume()
        rolling_volume = self.data['Volume'].rolling(window=self.time_step).sum().fillna(0)
        rolling_money_flow_volume = money_flow_volume.rolling(window=self.time_step).sum().fillna(0)
        cmf = rolling_money_flow_volume / rolling_volume
        return cmf
    
    def _close_ratio(self, window):
        money_flow_multiplier = self._money_flow_multiplier()
        close_ratio = money_flow_multiplier.rolling(window=window).mean().fillna(0)
        return close_ratio
    
    def _volume_percentile(self, window):
        rolling_volume = self.data['Volume'].rolling(window=window).mean().fillna(0)
        percentile = np.percentile(rolling_volume, np.arange(101))
        volume_percentile = pd.cut(rolling_volume, bins=percentile, labels=False, duplicates='drop').fillna(0)
        return volume_percentile
    
    def _price_change(self, data):
        changes = [0]
        for i in range(1, len(data)):
            if data[i - 1] == 0:
                changes.append(0)
            else:
                change = (data[i] - data[i - 1]) / data[i - 1] * 100
                changes.append(change)
        return changes   
 
    def _complete_data(self):
        # reset index
        new_idx = []
        for date in self.time_intervals:
            start_time = pd.to_datetime(date + ' 09:01:00')
            end_time = pd.to_datetime(date +' 13:30:00')
            datetime_range = pd.date_range(start_time, end_time, freq='T')
            new_idx += datetime_range
        self.data = self.data.reindex(new_idx)
        
        # fill missing values
        self.data['Open'] = self.data['Open'].fillna(method='ffill').interpolate(method='linear')
        self.data['High'] = self.data['High'].fillna(method='ffill').interpolate(method='linear')
        self.data['Low'] = self.data['Low'].fillna(method='ffill').interpolate(method='linear')
        self.data['Close'] = self.data['Close'].fillna(method='ffill').interpolate(method='linear')
        self.data['Volume'] = self.data['Volume'].fillna(method='ffill').interpolate(method='linear')
        self.data['Amount'] = self.data['Amount'].fillna(method='ffill').interpolate(method='linear')

class StockDataset(Dataset):
    def __init__(self, args, csv_file):
        # setting parameters
        self._args = args
        self.time_step = args.time_step # 每隔幾分鐘取出一筆資料
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
        
        self._standardize() 
        self._rolling_window()
    
    def _standardize(self):
        col_list = list(self.data.columns)[:6]
        if self._args.mode in ['train', 'optim']:
            # scaler_X = StandardScaler()
            # scaler_y = StandardScaler()
            scaler_X = MinMaxScaler(feature_range=[-1, 1])
            scaler_y = MinMaxScaler(feature_range=[-1, 1])
            self.data[col_list] = scaler_X.fit_transform(self.data[col_list].values)
            self.data['y'] = scaler_y.fit_transform(self.data[['y']].values)
            # save the scaler for testing purposes
            with open(f'./data/{self._args.stock}/scaler_X.pkl', 'wb') as f:
                pickle.dump(scaler_X, f)
            with open(f'./data/{self._args.stock}/scaler_y.pkl', 'wb') as f:
                pickle.dump(scaler_y, f)
        else:
            # load saved scaler
            with open(f'./data/{self._args.stock}/scaler_X.pkl', 'rb') as f:
                scaler_X = pickle.load(f)
            with open(f'./data/{self._args.stock}/scaler_y.pkl', 'rb') as f:
                scaler_y = pickle.load(f)
            self.data[col_list] = scaler_X.transform(self.data[col_list].values)
            if 'y' in self.data.columns:
                self.data['y'] = scaler_y.transform(self.data[['y']].values)
    
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
    data_processor = DataProcessor(args)