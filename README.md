# Environments
模擬資料需要有 R 環境，並且安裝 `rmgarch` 套件

# 檔案說明
## 資料夾
- `./data`: 真實股票數據、模擬數據
- `./lib`: 預處理、視覺化、計算指標……
- `./model`: 各模型訓練、測試程式碼、模型架構
## 檔案
- `arguments.py`: 參數設定
- `bayes_optim.py`: 進行貝氏優化
- `eval_simulation.py`:評估模型與模擬數據
- `eval.py`: 評估模型預測能力
- `main.py`: 執行預處理 -> `bayes_optim.py` -> `train.py` -> `eval.py`
- `simulated.py`: 產生模擬數據
- `strategy.py`: 產生交易訊號、交易策略
- `train.py`: 訓練模型