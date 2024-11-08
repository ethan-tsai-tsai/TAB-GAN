# Environments
模擬資料需要有 R 環境，並且安裝 `rmgarch` 套件

# 檔案說明
- `arguments.py`: 可調參數列表
- `bayes_optim.py`: 貝氏優化程式碼
- `eval.py`: 畫預測圖、預測資料
- `main.py`: 主程式，包含預處理、貝氏優化、訓練模型、預測值
- `model.py`: GAN 的 generator 和 discriminator
- `preprocessor.py`: 資料預處理、建立 Datasets
- `simulated.py`: 用 garch + prophet 建立模擬數據
- `train.py`: 訓練 WGAN-GP 模型
- `utils.py`: 視覺化、工具程式碼

# 指令
