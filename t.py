import numpy as np
import pandas as pd
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

def test_r_environment():
    """測試 R 環境是否正確設置"""
    try:
        pandas2ri.activate()
        rmgarch = importr('rmgarch')
        print("✓ R 環境設置成功")
        return True
    except Exception as e:
        print(f"✗ R 環境設置失敗: {str(e)}")
        return False

def test_garch_function():
    """測試 GARCH 模型函數"""
    # 創建測試數據
    np.random.seed(42)
    test_data = pd.DataFrame({
        'returns': np.random.normal(0, 1, 100)
    })
    
    try:
        # 轉換數據到 R 格式
        r_data = pandas2ri.py2rpy(test_data)
        
        # 定義 R 函數
        r_code = """
        function(data, n_sim) {
            spec = ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(1,1)),
                             mean.model=list(armaOrder=c(0,0)))
            fit = ugarchfit(spec, data[,1])
            sim = ugarchsim(fit, n.sim=n_sim)
            return sigma(sim)
        }
        """
        
        # 執行 R 函數
        r_func = robjects.r(r_code)
        errors = np.array(r_func(r_data, 50))
        
        if len(errors) == 50:
            print("✓ GARCH 模型執行成功")
            print(f"  - 生成的誤差項大小: {len(errors)}")
            print(f"  - 誤差項範例: {errors[:5]}")
            return True
    except Exception as e:
        print(f"✗ GARCH 模型執行失敗: {str(e)}")
        return False

def main():
    """主測試函數"""
    print("開始測試 R 整合功能...")
    print("-" * 50)
    
    # 測試 R 環境
    if not test_r_environment():
        print("環境測試失敗，終止後續測試")
        return
    
    print("-" * 50)
    
    # 測試 GARCH 函數
    test_garch_function()
    
    print("-" * 50)
    print("測試完成")

if __name__ == "__main__":
    main()