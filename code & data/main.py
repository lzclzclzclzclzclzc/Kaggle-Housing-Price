import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class HousePricePredictor:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.train_data = None
        self.test_data = None
        self.y = None
        self.X = None
        self.X_test = None
        self.model = None
        self.grid_search = None
        self.importance_df = None

        # === 挑选标准化个归一化的特征字段 === #
        self.numerical_features_standardize = [
            "LotFrontage", "LotArea", "YearBuilt", "YearRemodAdd", "MasVnrArea",
            "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF",
            "2ndFlrSF", "LowQualFinSF", "GrLivArea", "GarageYrBlt", "GarageArea",
            "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch",
            "PoolArea", "MiscVal", "MoSold", "YrSold"
        ]
        self.numerical_features_normalize = [
            "OverallQual", "OverallCond", "BsmtFullBath", "BsmtHalfBath",
            "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr",
            "TotRmsAbvGrd", "Fireplaces", "GarageCars"
        ]

    # === RMSE评分 === #
    def rmse(self, y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    # === 读取数据 === #
    def load_data(self):
        self.train_data = pd.read_csv(self.train_path)
        self.test_data = pd.read_csv(self.test_path)
        self.y = np.log1p(self.train_data["SalePrice"])
        self.train_data.drop("SalePrice", axis=1, inplace=True)

    # === 合并 train 和 test  === #
    def preprocess(self):
        all_data = pd.concat([self.train_data, self.test_data], axis=0, ignore_index=True)

        for col in all_data.columns:
            if all_data[col].dtype == "object":
                all_data[col] = all_data[col].fillna("None")
            else:
                all_data[col] = all_data[col].fillna(all_data[col].median())

        # === Label Encoding === #
        categoricals = all_data.select_dtypes(include="object").columns
        for col in categoricals:
            le = LabelEncoder()
            all_data[col] = le.fit_transform(all_data[col])

        # === 构建派生特征 === #
        all_data["TotalSF"] = all_data["TotalBsmtSF"] + all_data["1stFlrSF"] + all_data["2ndFlrSF"]
        all_data["TotalBath"] = (
            all_data["FullBath"] + 0.5 * all_data["HalfBath"] +
            all_data["BsmtFullBath"] + 0.5 * all_data["BsmtHalfBath"]
        )
        all_data["Age"] = all_data["YrSold"] - all_data["YearBuilt"]

        # === Standardization === #
        scaler_std = StandardScaler()
        all_data[self.numerical_features_standardize] = scaler_std.fit_transform(
            all_data[self.numerical_features_standardize]
        )

        # === Normalization === #
        scaler_minmax = MinMaxScaler()
        all_data[self.numerical_features_normalize] = scaler_minmax.fit_transform(
            all_data[self.numerical_features_normalize]
        )

        # Re-split
        self.X = all_data.iloc[:len(self.y), :]
        self.X_test = all_data.iloc[len(self.y):, :]

    # === 模型训练 === #
    def train_model(self):
        # === 定义参数网格 === #
        param_grid = {
            'n_estimators': list(range(100, 500, 50)),
            'max_depth': list(range(10, 30, 10)),
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'max_features': ['sqrt']
        }
        rf = RandomForestRegressor(random_state=1)
        scorer = make_scorer(self.rmse, greater_is_better=False)

        self.grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            scoring=scorer,
            cv=5,
            verbose=2,
            n_jobs=-1
        )
        self.grid_search.fit(self.X, self.y)
        self.model = self.grid_search.best_estimator_

        print("最佳参数组合:", self.grid_search.best_params_)
        print("最佳 RMSE:", -self.grid_search.best_score_)

    # ==== 计算 R² ==== #
    def evaluate(self):
        y_pred = self.model.predict(self.X)
        r2 = r2_score(self.y, y_pred)
        print(f"R² Score on Training Set: {r2:.4f}")

    # ==== 可视化所有参数对 RMSE 的影响 ==== #
    def plot_param_effects(self):
        results_df = pd.DataFrame(self.grid_search.cv_results_)
        results_df["rmse"] = -results_df["mean_test_score"]
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
    def plot_feature_importances(self, top_n=30):
        importances = self.model.feature_importances_
        feature_names = self.X.columns
        self.importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        self.importance_df = self.importance_df.sort_values(by="importance", ascending=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(x="importance", y="feature", data=self.importance_df.head(top_n))
        plt.title(f"Top {top_n} Feature Importances (Random Forest)")
        plt.tight_layout()
        plt.show()

    def predict_and_save(self, output_path='submission.csv'):
        preds = self.model.predict(self.X_test)
        output = pd.DataFrame({'Id': self.test_data["Id"], 'SalePrice': np.expm1(preds)})
        output.to_csv(output_path, index=False)
        print(f"预测完成，结果已保存为 {output_path}")

# ========== 使用示例 ========== #
if __name__ == "__main__":
    predictor = HousePricePredictor("train.csv", "test.csv")
    predictor.load_data()
    predictor.preprocess()
    predictor.train_model()
    predictor.evaluate()
    predictor.plot_param_effects()
    predictor.plot_feature_importances()
    predictor.predict_and_save()


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