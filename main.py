import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ==== RMSE评分 ==== #
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse, greater_is_better=False)

# ==== 读取数据 ==== #
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# ==== 取对数 ==== #
y = np.log1p(train["SalePrice"])
train.drop("SalePrice", axis=1, inplace=True)

# ==== 合并 train 和 test  ==== #
all_data = pd.concat([train, test], axis=0, ignore_index=True)

# ==== 补充缺失值 ==== #
for col in all_data.columns:
    if all_data[col].dtype == "object":
        all_data[col] = all_data[col].fillna("None")
    else:
        all_data[col] = all_data[col].fillna(all_data[col].median())

# ==== Label Encode ==== #
categoricals = all_data.select_dtypes(include="object").columns
for col in categoricals:
    le = LabelEncoder()
    all_data[col] = le.fit_transform(all_data[col])

# ==== 构建派生特征 ==== #
all_data["TotalSF"] = all_data["TotalBsmtSF"] + all_data["1stFlrSF"] + all_data["2ndFlrSF"]
all_data["TotalBath"] = (
    all_data["FullBath"] + 0.5 * all_data["HalfBath"] +
    all_data["BsmtFullBath"] + 0.5 * all_data["BsmtHalfBath"]
)
all_data["Age"] = all_data["YrSold"] - all_data["YearBuilt"]

# ==== 挑选标准化个归一化的特征字段 ==== #
numerical_features_standardize = [
    "LotFrontage",
    "LotArea",
    "YearBuilt",
    "YearRemodAdd",
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "LowQualFinSF",
    "GrLivArea",
    "GarageYrBlt",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "PoolArea",
    "MiscVal",
    "MoSold",
    "YrSold",
]

numerical_features_normalize = [
    "OverallQual",
    "OverallCond",
    "BsmtFullBath",
    "BsmtHalfBath",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "GarageCars",
]

# ==== 标准化 ==== #
scaler_standardize = StandardScaler()
all_data[numerical_features_standardize] = scaler_standardize.fit_transform(all_data[numerical_features_standardize])

# ==== 归一化 ==== #
scaler_normalize = MinMaxScaler()
all_data[numerical_features_normalize] = scaler_normalize.fit_transform(all_data[numerical_features_normalize])

# ==== 重新分成train 和 test ==== #
X = all_data.iloc[:len(y), :]
X_test = all_data.iloc[len(y):, :]

# ==== 使用的特征 ==== #
X = all_data.iloc[:len(train), :]  # 前面是训练集
X_test = all_data.iloc[len(train):, :]  # 后面是测试集

# ==== 定义参数网格 ==== #
param_grid_rf = {
    'n_estimators': list(range(100, 500, 50)),
    'max_depth': list(range(10, 30, 2)),
    'min_samples_split': list(range(2, 3, 1)),
    'min_samples_leaf': [1],
    'max_features': ['sqrt']
}

# === 网格搜索 + 10折交叉验证 === #
rf = RandomForestRegressor(random_state=1)
grid_search = GridSearchCV(estimator=rf,
                           param_grid=param_grid_rf,
                           scoring=rmse_scorer,
                           cv=10,
                           verbose=2,
                           n_jobs=-1)
grid_search.fit(X, y)

# ==== 结果转为 DataFrame ==== #
results_df = pd.DataFrame(grid_search.cv_results_)
results_df["rmse"] = -results_df["mean_test_score"]

# ==== 输出最佳结果 ==== #
print("最佳参数组合:", grid_search.best_params_)
print("最佳 RMSE (负数表示越小越好):", grid_search.best_score_)

# ==== 计算 R² ==== #
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X)
r2 = r2_score(y, y_pred)
print(f"R² Score on Training Set: {r2:.4f}")

# ==== 可视化所有参数对 RMSE 的影响 ==== #
param_names = [col for col in results_df.columns if col.startswith("param_")]

plt.figure(figsize=(20, 12))
for i, param in enumerate(param_names, 1):
    plt.subplot((len(param_names) + 1) // 2, 2, i)
    grouped = results_df.groupby(param)["rmse"].mean()
    grouped.plot(marker="o")
    plt.title(f"{param} vs. RMSE")
    plt.xlabel(param)
    plt.ylabel("Average RMSE")
    plt.grid(True)

plt.tight_layout()
plt.show()

# ==== 可视化特征重要性 ==== #
importances = best_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
importance_df = importance_df.sort_values(by="importance", ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x="importance", y="feature", data=importance_df.head(30))  # 前30个特征
plt.title("Top 30 Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()

# ==== 生成提交文件 ==== #
test_preds = best_model.predict(X_test)
output = pd.DataFrame({'Id': test.Id, 'SalePrice': np.expm1(test_preds)})
output.to_csv('submission.csv', index=False)
print("预测完成，结果已保存为 submission.csv")

# param_grid_rf = {
#     'n_estimators': list(range(100, 500, 10)),
#     'max_depth': list(range(10, 30, 2)),
#     'min_samples_split': list(range(2, 5, 1)),
#     'min_samples_leaf': [1],
#     'max_features': ['sqrt']
# }
# cv = 10
# 最佳参数组合: {'max_depth': 14, 'max_features': 'sqrt', 'min_sampes_leaf': 1, 'min_samples_split': 2, 'n_estimators': 260}
# 最佳 RMSE (负数表示越小越好): -0.13606095839513924l
# R² Score on Training Set: 0.9817

# param_grid_rf = {
#     'n_estimators': list(range(100, 500, 50)),
#     'max_depth': list(range(10, 30, 2)),
#     'min_samples_split': list(range(2, 5, 1)),
#     'min_samples_leaf': [1],
#     'max_features': ['sqrt']
# }
# cv = 10
# 最佳参数组合: {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
# 最佳 RMSE (负数表示越小越好): -0.13664915604809935
# R² Score on Training Set: 0.9833

# param_grid_rf = {
#     'n_estimators': list(range(100, 500, 10)),
#     'max_depth': list(range(10, 30, 2)),
#     'min_samples_split': list(range(2, 5, 1)),
#     'min_samples_leaf': [1],
#     'max_features': ['sqrt']
# }
# cv = 5
# 最佳参数组合: {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 300}
# 最佳 RMSE (负数表示越小越好): -0.1377026382815109
# R² Score on Training Set: 0.9810

# param_grid_rf = {
#     'n_estimators': [50, 100, 200],          # 树的数量：越多越稳，但训练更慢
#     'max_depth': [10, 20, 30],                # 树的最大深度：防止过拟合
#     'min_samples_split': [2, 5, 10],          # 节点分裂的最小样本数：越大越“保守”
#     'min_samples_leaf': [1, 2, 4],            # 叶子节点最小样本数：控制每棵树的复杂度
#     'max_features': ['sqrt', 'log2', None]    # 每次分裂考虑的特征数
# }