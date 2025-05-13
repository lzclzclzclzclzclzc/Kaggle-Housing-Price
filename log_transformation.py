import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取数据
train = pd.read_csv('train.csv')

# 2. 原始 SalePrice 偏度和峰度
original_skew = train["SalePrice"].skew()
original_kurt = train["SalePrice"].kurt()
ks_stat_orig, ks_pval_orig = stats.kstest(train["SalePrice"], 'norm', args=(train["SalePrice"].mean(), train["SalePrice"].std()))

print("原始 SalePrice 分析：")
print(f"偏度 (Skewness): {original_skew:.4f}")
print(f"峰度 (Kurtosis): {original_kurt:.4f}")
print(f"K-S 检验 p 值: {ks_pval_orig:.4f}\n")

# 可视化原始 SalePrice 分布
plt.figure(figsize=(8, 5))
sns.histplot(train["SalePrice"], kde=True)
plt.title("Original SalePrice Distribution")
plt.xlabel("SalePrice")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("original_saleprice.png")  # 保存图片
plt.close()  # 关闭图像，避免叠加

# 3. 对数变换
train["LogSalePrice"] = np.log1p(train["SalePrice"])

# 4. 对数变换后的偏度和峰度
log_skew = train["LogSalePrice"].skew()
log_kurt = train["LogSalePrice"].kurt()
ks_stat_log, ks_pval_log = stats.kstest(train["LogSalePrice"], 'norm', args=(train["LogSalePrice"].mean(), train["LogSalePrice"].std()))

print("对数化后的 SalePrice 分析：")
print(f"偏度 (Skewness): {log_skew:.4f}")
print(f"峰度 (Kurtosis): {log_kurt:.4f}")
print(f"K-S 检验 p 值: {ks_pval_log:.4f}")

# 可视化对数后的 SalePrice 分布
plt.figure(figsize=(8, 5))
sns.histplot(train["LogSalePrice"], kde=True)
plt.title("Log-transformed SalePrice Distribution")
plt.xlabel("Log(SalePrice)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("log_saleprice.png")  # 保存图片
plt.close()