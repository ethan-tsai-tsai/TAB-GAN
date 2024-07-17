import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from utils import *

class StockDataset(Dataset):
    def __init__(self, args, mode='train'):
        """
        data: csv: 股價資料的csv檔
        window_size: 取幾天的資料來預測下一天的資料
        target_length: 預測的資料長度
        time_step: 取出資料的時間間隔
        """
        # Caculate running time
        print('Processing data ......')
        start_time = datetime.now()
        
        # Load data
        self.data = pd.read_csv(f'./data/{args.stock}.csv').sort_values(by='ts')
        
        # 資料處理
        self.data['ts'] = pd.to_datetime(self.data['ts'])
        self.data.set_index(self.data['ts'], inplace=True)
        self.data = self.data.groupby(self.data.index).mean() # 重複日期
        
        # 取得參數
        self.time_step = args.time_step # 每隔幾分鐘取出一筆資料
        self.target_length = args.target_length // args.time_step # y的資料筆數 
        self.seq_len = args.window_size * (270 // args.time_step) # X的資料筆數
        self.window_size = args.window_size # 移動窗格大小
        self.window_stride = args.window_stride # 移動窗格的移動步伐
        
        # 補全缺失值
        self.time_intervals = self.data.index.strftime('%Y-%m-%d').unique().tolist() # 資料中的日期
        self.complete_data() 
        
        self.data['y'] = self.data['Close']
        self.standardize() # 正規化
        
        # 加入欄位
        self.data['7ma'] = EMA(self.data['Close'], 7)
        self.data['14ma'] = EMA(self.data['Close'], 14)
        self.data['21ma'] = EMA(self.data['Close'], 21)
        self.data['7macd'] = MACD(self.data['Close'], 3, 11, 7)
        self.data['14macd'] = MACD(self.data['Close'], 7, 21, 14)
        self.data['7rsi'] = RSI(self.data['Close'], 7)
        self.data['14rsi'] = RSI(self.data['Close'], 14)
        self.data['21rsi'] = RSI(self.data['Close'], 21)
        self.data['7atr'] = atr(self.data['High'], self.data['Low'], 7)
        self.data['14atr'] = atr(self.data['High'], self.data['Low'], 14)
        self.data['21atr'] = atr(self.data['High'], self.data['Low'], 21)
        self.data['7upper'], self.data['7lower'] = bollinger_band(self.data['Close'], 7)
        self.data['14upper'], self.data['14lower'] = bollinger_band(self.data['Close'], 14)
        self.data['21upper'], self.data['21lower'] = bollinger_band(self.data['Close'], 21)
        self.data['7rsv'] = rsv(self.data['Close'], 7)
        self.data['14rsv'] = rsv(self.data['Close'], 14)
        self.data['21rsv'] = rsv(self.data['Close'], 21)
        self.data = self.data.iloc[270::, :]
        # self.data['month'] = self.data.index.to_series().dt.month
        # self.data['day'] = self.data.index.to_series().dt.day
        # self.data['hour'] = self.data.index.to_series().dt.hour
        # self.data['minute'] = self.data.index.to_series().dt.minute
        
        if mode == 'train':
            self.rolling_window() # 移動窗格
        
        # 取得資訊
        self.num_features = len(self.data.drop('y', axis=1).columns) # 特徵數量
        
        # 防呆
        if self.data.isnull().values.any():
            print('There are missing values in the data.')
            null_columns = self.data.columns[self.data.isnan().any()]
            null_data = self.data[null_columns]
            print(null_data)
            os._exit(0)

        if np.isinf(self.data.values).any():
            print('There are inf values in the data.')
            inf_rows = self.data.columns[np.isinf(self.data).any()]
            inf_data = self.data[inf_rows]
            print(inf_data)
            os._exit(0)
        end_time = datetime.now()
        print(f'Data processing spent {(end_time - start_time).total_seconds(): 2f} seconds')
        
    def standardize(self):
        self.scaler_X = MinMaxScaler(feature_range=[-1, 1])
        self.scaler_y = MinMaxScaler(feature_range=[-1, 1])
        col_list = list(self.data.columns)
        col_list.remove('y')
        self.data[col_list] = self.scaler_X.fit_transform(self.data[col_list].values)
        self.data['y'] = self.scaler_y.fit_transform(self.data[['y']].values)
    
    def complete_data(self):
        new_idx = []
        for date in self.time_intervals:
            start_time = pd.to_datetime(date + ' 09:01:00')
            end_time = pd.to_datetime(date +' 13:30:00')
            datetime_range = pd.date_range(start_time, end_time, freq='T')
            new_idx += datetime_range
        self.data = self.data.reindex(new_idx)
        
        # 計算填補值
        self.data['MA_Open'] = self.data['Open'].rolling(window=5, min_periods=1).mean()
        self.data['MA_High'] = self.data['High'].rolling(window=5, min_periods=1).mean()
        self.data['MA_Low'] = self.data['Low'].rolling(window=5, min_periods=1).mean()
        self.data['MA_Close'] = self.data['Close'].rolling(window=5, min_periods=1).mean()
        self.data['MA_Volume'] = self.data['Volume'].rolling(window=5, min_periods=1).mean()
        self.data['MA_Amount'] = self.data['Amount'].rolling(window=5, min_periods=1).mean()

        # 填補遺失值
        self.data['Open'] = self.data['Open'].fillna(self.data['MA_Open'])
        self.data['High'] = self.data['High'].fillna(self.data['MA_High'])
        self.data['Low'] = self.data['Low'].fillna(self.data['MA_Low'])
        self.data['Close'] = self.data['Close'].fillna(self.data['MA_Close'])
        self.data['Volume'] = self.data['Volume'].fillna(self.data['MA_Volume'])
        self.data['Amount'] = self.data['Amount'].fillna(self.data['MA_Amount'])
        self.data = self.data.drop(['MA_Open', 'MA_High', 'MA_Low', 'MA_Close', 'MA_Volume', 'MA_Amount'], axis=1)
        # 遞迴
        if self.data.isnull().values.any():
            print(f'Data still has {self.data.isnull().sum().sum()} missing value, try again complete data.')
            self.complete_data()
        
    def rolling_window(self):
        self.data = self.data.iloc[::self.time_step, :]
        self.X, self.y = [], []
        for i in range(0, len(self.data) - self.seq_len - self.target_length, self.window_stride):
            self.X.append(self.data.iloc[i:i+self.seq_len, :len(self.data.columns)-1].values)
            self.y.append(self.data.iloc[i+self.seq_len:i+self.seq_len+self.target_length, len(self.data.columns)-1].values) # 只取最後
            # self.y.append(self.data.iloc[i:i+self.seq_len+self.target_length, len(self.data.columns)-1].values) # 取X天數+最後

    def get_data(self, date, days):
        X, y= [], []
        date = pd.to_datetime(date + ' 09:01:00')
        try:
            idx = self.data.index.get_loc(date)
            for i in range(idx, 1+idx + (days-1)*self.seq_len//self.window_size, self.window_stride):
                X.append(self.data.iloc[i:i+self.seq_len, :len(self.data.columns)-1].values)
                y.append(self.data.iloc[i+self.seq_len:i+self.seq_len+self.target_length, len(self.data.columns)-1].values) # 只取最後
                # y.append(self.data.iloc[i:i+self.seq_len+self.target_length, len(self.data.columns)-1].values) # 取X天數+最後
        except:
            idx = len(self.data) - self.seq_len//self.window_size
            X.append(self.data.iloc[idx-self.seq_len:idx, :].values)
            y.append(self.data.iloc[idx: idx+self.target_length, 3].values)
        X = np.array(X)
        y = np.array(y)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)