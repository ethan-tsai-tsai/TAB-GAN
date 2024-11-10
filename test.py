import numpy as np
import pandas as pd
from rpy2.robjects import pandas2ri, r
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import warnings
warnings.filterwarnings('ignore')

class DCCGARCHTester:
    def __init__(self):
        # 初始化 R 環境
        pandas2ri.activate()
        self.rmgarch = importr('rmgarch')
        self.rugarch = importr('rugarch')
        self.stats = importr('stats')
        self.base = importr('base')
        
    def generate_test_data(self, n_series=3, n_obs=1000):
        """生成測試用的模擬數據"""
        np.random.seed(42)
        # 生成相關的隨機數據
        cov_matrix = np.array([[1, 0.5, 0.3],
                              [0.5, 1, 0.4],
                              [0.3, 0.4, 1]])
        data = np.random.multivariate_normal(
            mean=np.zeros(n_series),
            cov=cov_matrix,
            size=n_obs
        )
        return pd.DataFrame(
            data, 
            columns=[f'Series_{i+1}' for i in range(n_series)]
        )
    
    def test_univariate_garch_spec(self, data):
        """測試單變量 GARCH 規格"""
        print("\nTesting univariate GARCH specification...")
        
        r_code = """
        function(data) {
            library(rmgarch)
            tryCatch({
                # 創建單變量 GARCH 規格
                spec = ugarchspec(
                    mean.model = list(armaOrder = c(1,1)),
                    variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
                    distribution.model = "std"
                )
                
                # 使用第一列數據進行擬合測試
                fit = ugarchfit(spec, data[,1])
                
                return(list(success = TRUE, message = "Univariate GARCH specification successful"))
            }, error = function(e) {
                return(list(success = FALSE, message = paste("Error:", e$message)))
            })
        }
        """
        
        r_test = ro.r(r_code)
        result = r_test(pandas2ri.py2rpy(data))
        
        print(f"Success: {result.rx2('success')[0]}")
        print(f"Message: {result.rx2('message')[0]}")
        
    def test_multispec(self, data):
        """測試多變量規格"""
        print("\nTesting multispec creation...")
        
        r_code = """
        function(data) {
            library(rmgarch)
            tryCatch({
                # 創建多變量規格
                uspec = multispec(
                    replicate(ncol(data),
                        ugarchspec(
                            mean.model = list(armaOrder = c(1,1)),
                            variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
                            distribution.model = "std"
                        )
                    )
                )
                
                return(list(success = TRUE, message = "Multispec creation successful"))
            }, error = function(e) {
                return(list(success = FALSE, message = paste("Error:", e$message)))
            })
        }
        """
        
        r_test = ro.r(r_code)
        result = r_test(pandas2ri.py2rpy(data))
        
        print(f"Success: {result.rx2('success')[0]}")
        print(f"Message: {result.rx2('message')[0]}")
    
    def test_dcc_spec(self, data):
        """測試 DCC 規格"""
        print("\nTesting DCC specification...")
        
        r_code = """
        function(data) {
            library(rmgarch)
            tryCatch({
                # 創建多變量規格
                uspec = multispec(
                    replicate(ncol(data),
                        ugarchspec(
                            mean.model = list(armaOrder = c(1,1)),
                            variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
                            distribution.model = "std"
                        )
                    )
                )
                
                # 創建 DCC 規格
                dccspec = dccspec(
                    uspec = uspec,
                    dccOrder = c(1,1),
                    distribution = "mvnorm"
                )
                
                return(list(success = TRUE, message = "DCC specification successful"))
            }, error = function(e) {
                return(list(success = FALSE, message = paste("Error:", e$message)))
            })
        }
        """
        
        r_test = ro.r(r_code)
        result = r_test(pandas2ri.py2rpy(data))
        
        print(f"Success: {result.rx2('success')[0]}")
        print(f"Message: {result.rx2('message')[0]}")
    
    def test_dcc_fit(self, data):
        """測試 DCC 擬合"""
        print("\nTesting DCC fitting...")
        
        r_code = """
        function(data) {
            library(rmgarch)
            tryCatch({
                # 創建多變量規格
                uspec = multispec(
                    replicate(ncol(data),
                        ugarchspec(
                            mean.model = list(armaOrder = c(1,1)),
                            variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
                            distribution.model = "std"
                        )
                    )
                )
                
                # 創建 DCC 規格
                dccspec = dccspec(
                    uspec = uspec,
                    dccOrder = c(1,1),
                    distribution = "mvnorm"
                )
                
                # 擬合模型
                fit = dccfit(dccspec, data = data, solver = "solnp")
                
                return(list(
                    success = TRUE, 
                    message = "DCC fitting successful",
                    convergence = fit@mfit$convergence
                ))
            }, error = function(e) {
                return(list(
                    success = FALSE, 
                    message = paste("Error:", e$message),
                    convergence = NA
                ))
            })
        }
        """
        
        r_test = ro.r(r_code)
        result = r_test(pandas2ri.py2rpy(data))
        
        print(f"Success: {result.rx2('success')[0]}")
        print(f"Message: {result.rx2('message')[0]}")
        if result.rx2('convergence')[0] is not ro.NA_Real:
            print(f"Convergence: {result.rx2('convergence')[0]}")
    
    def test_dcc_sim(self, data, n_ahead=10):
        """測試 DCC 模擬"""
        print("\nTesting DCC simulation...")
        
        r_code = """
        function(data, n_ahead) {
            library(rmgarch)
            tryCatch({
                # 創建和擬合模型
                uspec = multispec(
                    replicate(ncol(data),
                        ugarchspec(
                            mean.model = list(armaOrder = c(1,1)),
                            variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
                            distribution.model = "std"
                        )
                    )
                )
                
                dccspec = dccspec(
                    uspec = uspec,
                    dccOrder = c(1,1),
                    distribution = "mvnorm"
                )
                
                fit = dccfit(dccspec, data = data, solver = "solnp")
                
                # 執行模擬
                sim = dccsim(fit, n.sim = n_ahead)
                
                return(list(
                    success = TRUE,
                    message = "DCC simulation successful",
                    sim_data = fitted(sim)
                ))
            }, error = function(e) {
                return(list(
                    success = FALSE,
                    message = paste("Error:", e$message),
                    sim_data = NA
                ))
            })
        }
        """
        
        r_test = ro.r(r_code)
        result = r_test(pandas2ri.py2rpy(data), n_ahead)
        
        print(f"Success: {result.rx2('success')[0]}")
        print(f"Message: {result.rx2('message')[0]}")
        if result.rx2('sim_data')[0] is not ro.NA_Real:
            sim_data = np.array(result.rx2('sim_data'))
            print(f"Simulated data shape: {sim_data.shape}")

    def run_all_tests(self):
        """執行所有測試"""
        print("Generating test data...")
        data = self.generate_test_data()
        print(f"Test data shape: {data.shape}")
        
        self.test_univariate_garch_spec(data)
        self.test_multispec(data)
        self.test_dcc_spec(data)
        self.test_dcc_fit(data)
        self.test_dcc_sim(data)

if __name__ == '__main__':
    tester = DCCGARCHTester()
    tester.run_all_tests()