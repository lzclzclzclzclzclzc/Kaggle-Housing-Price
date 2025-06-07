https://www.kaggle.com/competitions/home-data-for-ml-course
SHANGHAI UNIVERSITY

课程设计（论文）

**UNDERGRADUATE COURSE (THESIS)**

**题目:基于随机森林算法的房价预测**


# 目录

[摘要 III](#_Toc198135148)

[ABSTRACT IV](#_Toc198135149)

[第1章 绪论 1](#_Toc198135150)

[§1.1 研究背景 1](#_Toc198135151)

[§1.2 本文研究内容及目标 1](#_Toc198135152)

[§1.2.1 研究内容 1](#_Toc198135153)

[§1.2.2 研究目标 1](#_Toc198135154)

[§1.3 本文组织结构 2](#_Toc198135155)

[第2章 数据集的采集 3](#_Toc198135156)

[§2.1 数据来源 3](#_Toc198135157)

[§2.2 数据集描述 3](#_Toc198135158)

[§2.2.1 总括 3](#_Toc198135159)

[§2.2.2 特征变量描述 3](#_Toc198135160)

[第3章 数据预处理 8](#_Toc198135161)

[§3.1 数据读取与目标变量变换 8](#_Toc198135162)

[§3.2 数据合并与缺失值处理 9](#_Toc198135163)

[§3.3 特征编码 9](#_Toc198135164)

[§3.3.1 特征编码的作用 9](#_Toc198135165)

[§3.3.2 与独热编码对比 10](#_Toc198135166)

[§3.4 派生特征构造 10](#_Toc198135167)

[§3.5 数据特征标准化和归一化 10](#_Toc198135168)

[§3.5.1 标准化 11](#_Toc198135169)

[§3.5.2 归一化 11](#_Toc198135170)

[§3.6 章节小结 12](#_Toc198135171)

[第4章 模型构建与参数优化 13](#_Toc198135172)

[§4.1 模型算法介绍 13](#_Toc198135173)

[§4.2 参数调整 13](#_Toc198135174)

[§4.3 测试集预测及提交结果 15](#_Toc198135175)

[§4.4 本章小结 15](#_Toc198135176)

[第5章 数据可视化 16](#_Toc198135177)

[§5.1 不同参数的变化对于RMSE的敏感度 16](#_Toc198135178)

[§5.2 特征重要性分析 17](#_Toc198135179)

[第6章 总结与展望 18](#_Toc198135180)

[§6.1 本文总结 18](#_Toc198135181)

[§6.1.1 本文的主要工作 18](#_Toc198135182)

[§6.2 局限性 18](#_Toc198135183)

[§6.3 改进方向 19](#_Toc198135184)

[致谢 20](#_Toc198135185)

[参考文献 21](#_Toc198135186)

[附录：源程序 23](#_Toc198135187)

基于随机森林算法的房价预测

# 摘要

本实验基于Kaggle提供的“Ames Housing”房价预测竞赛数据集，构建了一个随机森林算法的回归模型，来预测Ames市住宅的最终销售价格。Ames Housing数据集包含79个特征变量，全面描述了房屋的结构、设施、装修和地理信息，为房价建模提供了丰富的基础。

为提高模型的泛化能力与解释性，本实验采用了面向对象的方式，实现了数据预处理、特征工程、建模、评估与预测的过程。数据预处理阶段包括缺失值填充、类别变量编码、特征标准化与归一化，并构造了派生特征。实验在模型训练时使用了带网格搜索的随机森林回归器，并以RMSE作为评价指标进行了5及10折交叉验证，从多个参数组合中选取最优模型。结果显示该模型在训练集上表现良好，R²得分高于0.95。将预测测试集的结果提交到kaggle网站上后，最佳得分为15914.50449。

为了更好更清晰地理解模型行为与数据特征，实验还通过可视化手段分析了模型在不同参数设置下的RMSE变化趋势，以及最重要的30个特征的重要性得分，帮助揭示模型预测依据与影响房价的关键因素。

关键词：特征工程；随机森林；网格搜索；回归分析；机器学习。

House Price Prediction Based on the Random Forest Algorithm

# ABSTRACT

This research is based on the "Ames Housing" dataset from a Kaggle competition. A regression model using the Random Forest algorithm was built to predict the final sale price of houses in Ames. The Ames Housing dataset contains 79 feature variables, whose features has described the house structure, facilities, decoration, and location, which has provided a strong base for price prediction.

To improve generalization and interpretability, the research used an object-oriented programming approach. The process included data preprocessing, feature engineering, model training, evaluation, and prediction. In data preprocessing, missing values were filled, categorical variables were encoded, and features were standardized and normalized. Some new features were also created. For training, the research used a Random Forest Regressor with grid search. RMSE was used as the evaluation metric, and both 5-fold and 10-fold cross-validation were applied. The best model achieved good results on the training set, with an R² score above 0.95. The best score on the Kaggle test set was 15914.50449.

To better understand the model and data, the research used visual analysis. Both RMSE changes under different parameter settings and the top 30 most important features were visualized. This helped explain how the model works and which features most affect house prices.

**Keywords:** Feature engineering; Random forest; Grid search; Regression analysis; Machine learning.

# 绪论

## 研究背景

在现实生活中，房价不仅受房屋面积、卧室数量等显性指标的影响，还受到房屋结构、装修质量、地理位置、周边环境乃至销售时间等多种因素的综合作用。传统线性回归难以充分刻画这些复杂非线性关系，因此越来越多的研究采用机器学习方法进行房价预测，以提升模型的准确性与鲁棒性。

本实验基于Kaggle提供的“Ames Housing”房价预测竞赛数据集，该数据集由Dean De Cock整理，涵盖美国爱荷华州Ames市的住宅样本，并提供了79个用于描述房屋各方面属性的特征变量。这些特征几乎囊括了房屋销售中影响价格的所有可能因素，是房价建模与预测的理想数据来源。

## 本文研究内容及目标

### 研究内容

本实验采用面向对象编程（OOP）**方法**，对房价预测的完整流程进行了模块化实现。具体内容包括：

（1）数据清洗与预处理；

（2）特征工程；

（3）建模与调参；

（4）模型预测。

（5）性能评估；

（6）数据可视化；

### 研究目标

本实验旨在构建一个鲁棒性强、解释性好且泛化能力优良的房价预测模型，具体目标包括：

（1）利用随机森林等集成学习方法有效建模高维住宅特征与房价之间的复杂关系；

（2）通过科学的数据预处理与特征工程手段提高模型精度与可解释性；

（3）探索不同交叉验证策略（如5折与10折）对模型调参结果的影响；

（4）利用参数可视化与特征重要性图揭示影响房价的关键因素.

## 本文组织结构

整篇论文分为五章。

第一章介绍了研究背景，并提出了本文的研究内容以及研究目标。

第二章主要介绍了数据集的来源，并描述了数据集的具体内容。

第三章首先介绍了数据预处理的具体过程，包括缺失值填充、类别变量编码、特征标准化与归一化、构造派生特征。

第四章主要描述了本实验采用的随机森林回归模型，以及参数调优的过程。

第五章展示了数据可视化的结果，包括RMSE对各个模型参数的敏感度和贡献度最高的30个特征。

第六章对全文进行了总结，归纳了本文的主要工作，并指出了缺陷和进一步研究的方向。

# 数据集的采集

本章具体描述了数据集的来源，同时对数据集的各特征字段进行描述。

## 数据来源

本实验所采用的数据集是来自kaggle.com网站上的Housing Prices Competition。其中包括训练集train.csv、测试集test.csv、基于销售年份和月份、地块面积和卧室数量的线性回归基准提交结果样例sample_submission.csv和各个特征字段的具体描述data_description.txt（原作者是Dean De Cock，但kaggle.com网站对数据集进行了轻微的修改来匹配这里用到的列名称）。

## 数据集描述

### 总括

训练集train.csv和测试集test.csv中囊括了包含训练集（train.csv）和测试集（test.csv）两个部分。数据集详细记录了美国爱荷华州埃姆斯市（Ames, Iowa）住宅的79个特征变量，涵盖建筑结构、区位特征、装修状况等多维度信息。训练集样本量为1,460条观测值，除79个特征变量外，还包含目标变量——房屋最终成交价格（SalePrice，单位：美元）。测试集样本量为1,459条观测值，仅包含特征变量。

提交文件的格式规范由示例文件sample_submission.csv明确规定。最终提交的预测结果文件（submission.csv）应包含两列数据：第一列为房屋编号（Id），第二列为模型预测的房屋售价（SalePrice）。

### 特征变量描述

训练集和测试集的特征变量的具体描述如下表所示，每个特征变量的不同取值的描述见附录。

表2.1 特征变量描述

| **变量名** | **英文描述** | **中文描述** |
| --- | --- | --- |
| **SalePrice** | The property's sale price in dollars | 房屋销售价格（美元） |
| **MSSubClass** | The building class | 建筑类别 |
| **MSZoning** | The general zoning classification | 区域规划分类 |
| **LotFrontage** | Linear feet of street connected to property | 临街长度（线性英尺） |
| **LotArea** | Lot size in square feet | 地块面积（平方英尺） |
| **Street** | Type of road access | 道路类型 |
| **Alley** | Type of alley access | 巷道路径类型 |
| **LotShape** | General shape of property | 地块形状 |
| **LandContour** | Flatness of the property | 地块平坦度 |
| **Utilities** | Type of utilities available | 公用设施类型 |
| **LotConfig** | Lot configuration | 地块配置 |
| **LandSlope** | Slope of property | 地块坡度 |
| **Neighborhood** | Physical locations within Ames city limits | 社区位置 |
| **Condition1** | Proximity to main road or railroad | 邻近主干道或铁路条件 |
| **Condition2** | Proximity to main road or railroad (if a second is present) | 邻近主干道或铁路附加条件 |
| **BldgType** | Type of dwelling | 住宅类型 |
| **HouseStyle** | Style of dwelling | 住宅风格 |
| **OverallQual** | Overall material and finish quality | 房屋整体质量评分 |
| **OverallCond** | Overall condition rating | 房屋整体状况评分 |
| **YearBuilt** | Original construction date | 建造年份 |
| **YearRemodAdd** | Remodel date | 改建年份 |
| **RoofStyle** | Type of roof | 屋顶类型 |
| **RoofMatl** | Roof material | 屋顶材料 |
| **Exterior1st** | Exterior covering on house | 主要外墙材料 |
| **Exterior2nd** | Exterior covering on house (if more than one material) | 次要外墙材料 |
| **MasVnrType** | Masonry veneer type | 砌体饰面类型 |
| **MasVnrArea** | Masonry veneer area in square feet | 砌体饰面面积（平方英尺） |
| **ExterQual** | Exterior material quality | 外墙材料质量 |
| **ExterCond** | Present condition of the material on the exterior | 外墙材料现状 |
| **Foundation** | Type of foundation | 地基类型 |
| **BsmtQual** | Height of the basement | 地下室高度质量 |
| **BsmtCond** | General condition of the basement | 地下室整体状况 |
| **BsmtExposure** | Walkout or garden level basement walls | 地下室采光水平 |
| **BsmtFinType1** | Quality of basement finished area | 地下室装修类型1 |
| **BsmtFinSF1** | Type 1 finished square feet | 类型1装修面积（平方英尺） |
| **BsmtFinType2** | Quality of second finished area (if present) | 地下室装修类型2 |
| **BsmtFinSF2** | Type 2 finished square feet | 类型2装修面积（平方英尺） |
| **BsmtUnfSF** | Unfinished square feet of basement area | 地下室未装修面积（平方英尺） |
| **TotalBsmtSF** | Total square feet of basement area | 地下室总面积（平方英尺） |
| **Heating** | Type of heating | 供暖类型 |
| **HeatingQC** | Heating quality and condition | 供暖质量 |
| **CentralAir** | Central air conditioning | 中央空调 |
| **Electrical** | Electrical system | 电力系统 |
| **1stFlrSF** | First Floor square feet | 首层面积（平方英尺） |
| **2ndFlrSF** | Second floor square feet | 二层面积（平方英尺） |
| **LowQualFinSF** | Low quality finished square feet (all floors) | 低质装修面积（平方英尺） |
| **GrLivArea** | Above grade (ground) living area square feet | 地面以上居住面积（平方英尺） |
| **BsmtFullBath** | Basement full bathrooms | 地下室全浴室数量 |
| **BsmtHalfBath** | Basement half bathrooms | 地下室半浴室数量 |
| **FullBath** | Full bathrooms above grade | 地面以上全浴室数量 |
| **HalfBath** | Half baths above grade | 地面以上半浴室数量 |
| **Bedroom** | Number of bedrooms above basement level | 地面以上卧室数量 |
| **Kitchen** | Number of kitchens | 厨房数量 |
| **KitchenQual** | Kitchen quality | 厨房质量 |
| **TotRmsAbvGrd** | Total rooms above grade (does not include bathrooms) | 地面以上总房间数（不含浴室） |
| **Functional** | Home functionality rating | 房屋功能性评级 |
| **Fireplaces** | Number of fireplaces | 壁炉数量 |
| **FireplaceQu** | Fireplace quality | 壁炉质量 |
| **GarageType** | Garage location | 车库位置类型 |
| **GarageYrBlt** | Year garage was built | 车库建造年份 |
| **GarageFinish** | Interior finish of the garage | 车库内部装修 |
| **GarageCars** | Size of garage in car capacity | 车库可容纳车辆数 |
| **GarageArea** | Size of garage in square feet | 车库面积（平方英尺） |
| **GarageQual** | Garage quality | 车库质量 |
| **GarageCond** | Garage condition | 车库状况 |
| **PavedDrive** | Paved driveway | 车道铺装情况 |
| **WoodDeckSF** | Wood deck area in square feet | 木质平台面积（平方英尺） |
| **OpenPorchSF** | Open porch area in square feet | 开放式门廊面积（平方英尺） |
| **EnclosedPorch** | Enclosed porch area in square feet | 封闭式门廊面积（平方英尺） |
| **3SsnPorch** | Three season porch area in square feet | 三季门廊面积（平方英尺） |
| **ScreenPorch** | Screen porch area in square feet | 纱窗门廊面积（平方英尺） |
| **PoolArea** | Pool area in square feet | 泳池面积（平方英尺） |
| **PoolQC** | Pool quality | 泳池质量 |
| **Fence** | Fence quality | 围栏质量 |
| **MiscFeature** | Miscellaneous feature not covered in other categories | 其他未分类设施 |
| **MiscVal** | $Value of miscellaneous feature | 其他设施价值（美元） |
| **MoSold** | Month Sold | 销售月份 |
| **YrSold** | Year Sold | 销售年份 |
| **SaleType** | Type of sale | 销售类型 |
| **SaleCondition** | Condition of sale | 销售条件 |

# 数据预处理

## 数据读取与目标变量变换

研究采用Kaggle平台提供的Ames住房数据集，包含训练集（train.csv，1,460条样本）和测试集（test.csv，1,459条样本）。数据加载后，首先对目标变量房屋销售价格（SalePrice）进行分布分析。通过Kolmogorov-Smirnov检验发现其呈现显著右偏分布（偏度=1.8829，峰度=6.5363，K-S检验p值为0.0000），这与城市房地产市场中高价房分布稀疏的特征一致。

图3-1原始SalePrice分布

为缓解偏态分布对回归模型的影响，采用自然对数变换进行数据转换（y = np.log1p(train\["SalePrice"\])。此方法不仅使目标变量更接近正态分布（偏度降至0.1213，峰度降至0.8095，K-S 检验 p 值增加到0.0147），还能降低极端值对损失函数的干扰，提升模型的鲁棒性。

图3-2对数化后SalePrice分布

随后将原SalePrice列从训练集中移除，以避免在特征工程阶段引入数据泄漏风险。

## 数据合并与缺失值处理

为确保训练集与测试集在预处理流程中的一致性，采用纵向合并策略（pd.concat(axis=0)）生成包含2,919条样本的整合数据集。缺失值处理采用分层填充策略：

（1）对于字符串（object）类型字段，将缺失值统一填充为 "None"；

（2）对于数值类型字段，使用该列的中位数进行填充.相较于均值填充，中位数对异常值更具抵抗力，尤其适用于存在长尾分布的变量。

这种处理方法既简洁又能有效保留样本信息，适用于树模型对异常值不敏感的特性。

## 特征编码

在机器学习建模中，特征编码是将非数值型分类变量转化为数值型表示的核心步骤，用来适配模型。标签编码通过单列整数值替代原始字符串，降低了内存占用与计算复杂度。同时，编码后的特征重要性可通过模型（如随机森林的Gini重要性）量化，明确不同类别对预测结果的贡献度。

### 特征编码的作用

大多数机器学习算法（如线性回归、支持向量机、神经网络等）的数学基础依赖于数值运算，无法直接处理文本或符号形式的分类变量（如MSZoning中的"RL"、"RM"等类别）。通过标签编码（Label Encoding）或独热编码（One-Hot Encoding），可将离散的类别信息转化为数值形式，使模型能够解析类别间的潜在模式。

### 与独热编码对比

本研究中采用标签编码而非独热编码，主要基于以下考量：

1.  **计算效率**：独热编码会显著增加特征维度（如25个社区的Neighborhood将扩展为25列），导致高维稀疏矩阵，加剧维度灾难（Curse of Dimensionality），影响树模型的计算效率。
2.  **模型特性适配：随机森林等树模型通过递归分割特征空间进行分类，其分裂规则对编码后的序数特征具有鲁棒性。即使标签编码引入非真实序数关系（如MSZoning中"RL"→3与"RM"→4并无实际大小意义），树模型仍能通过多次分裂恢复类别间差异。**

## 派生特征构造

在机器学习建模中，**派生特征构造**（Feature Engineering）是通过组合、转换或提取现有特征生成新特征的过程，其核心目标是从原始数据中挖掘潜在信息，增强模型对复杂模式的捕捉能力。在房价预测任务中，本研究通过领域知识构建了三个关键派生特征（TotalSF、TotalBath、Age）：

1.  TotalSF**（总使用面积）**：整合地下室与各楼层面积（TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF），反映房屋整体空间规模。该特征与房价呈显著正相关。
2.  TotalBath（**等效浴室数）：通过加权求和（**TotalBath = FullBath + 0.5 × HalfBath + BsmtFullBath + 0.5 × BsmtHalfBath**）量化卫浴设施的功能完整性。**
3.  Age**（房龄）：计算为销售年份与建造年份之差（**Age = YrSold − YearBuilt**），直接表征建筑折旧程度。房龄与房价呈非线性负相关。**

原始特征往往存在冗余或隐含关联，直接建模可能导致多重共线性或信息重叠。派生特征通过解耦关键变量，增强特征的语义表达能力。同时，派生特征能显式编码领域知识，帮助模型捕捉原始数据中隐含的非线性效应，抵抗数据噪声，增强模型泛化能力。

## 数据特征标准化和归一化

在机器学习建模中，**特征尺度一致性**是优化模型性能的核心前提。不同量纲或量级的特征会导致梯度下降算法收敛速度降低、距离度量失真（如欧氏距离计算时大范围特征主导结果），甚至引发数值计算溢出问题。为此，本研究基于特征的数据分布特性与物理意义，采用差异化尺度变换策略——标准化（Standardization）与归一化（Normalization）。

### 标准化

定义与公式：标准化通过对特征列进行线性变换，使其服从均值为0、标准差为1的正态分布：

其中 _μ_ 为特征均值，_σ_ 为标准差。

标准化适用于具有明显量纲单位、且数值波动范围较大或偏态显著的变量，如面积、建造年份、地下室和车库面积等。这些变量本身在原始尺度上数值较大（例如 LotArea 可能达到几万），若直接用于模型训练可能导致距离度量偏斜，影响模型性能。

进行标准化的特征字段包括：  
"LotFrontage"、"LotArea"、"YearBuilt"、"YearRemodAdd"、"MasVnrArea"、 "BsmtFinSF1"、"BsmtFinSF2"、"BsmtUnfSF"、"TotalBsmtSF"、"1stFlrSF"、"2ndFlrSF"、"LowQualFinSF"、"GrLivArea"、"GarageYrBlt"、"GarageArea"、"WoodDeckSF"、"OpenPorchSF"、"EnclosedPorch"、"3SsnPorch"、"ScreenPorch"、"PoolArea"、"MiscVal"、"MoSold"、"YrSold"。

### 归一化

定义与公式：归一化将特征线性映射至固定区间（通常为\[0,1\]）：

对取值有限且具有等级意义的离散型变量（如评分、计数类特征），归一化可维持原始比例关系，避免标准化引入的负值干扰解释性。归一化适用于本身取值范围有限、且对比例差异敏感的评分类或计数类变量，如房屋综合评级（OverallQual）、卧室数量（BedroomAbvGr）、车库车位数（GarageCars）等。这类变量往往是离散整数型，取值分布较集中特定区间，如 0–10、1–5 等，标准化后可能会放大无意义的波动，因此更适合归一化处理以保留其等级间相对关系。

进行归一化的特征字段包括：

"OverallQual"、"OverallCond"、"BsmtFullBath"、"BsmtHalfBath"、"FullBath"、"HalfBath"、"BedroomAbvGr"、"KitchenAbvGr"、"TotRmsAbvGrd"、"Fireplaces"、"GarageCars"。

综上，基于变量的分布特性和语义属性划分处理方式，不仅提高了模型对输入特征的学习效率，也增强了模型的数值稳定性和泛化能力。

## 章节小结

本章围绕房价预测任务的数据预处理流程进行了系统而详尽的阐述，构成了整个建模过程的核心基础。

首先对目标变量SalePrice进行了分布分析与对数变换，有效缓解了偏态分布对回归模型的不利影响，提高了模型的稳健性与预测准确性。并通过训练集与测试集的合并处理，统一了特征工程流程，并采用分类型与数值型变量分层处理的策略，有效解决了缺失值问题，确保了数据完整性与一致性。

在特征编码阶段，采用标签编码代替独热编码，充分考虑模型的计算效率与结构特点，减少了维度冗余问题，同时保留了分类变量的信息表达能力。随后通过构建三个具有强解释力的派生特征（TotalSF、TotalBath、Age），进一步增强了模型对数据中隐含模式的建模能力。

本实验依据变量的分布特性与语义属性，选择分别采用标准化与归一化技术对特征进行了差异化尺度变换，不仅改善了特征间的尺度差异，提升了模型的数值稳定性，也为后续建模过程打下了坚实基础。

综上所述，本章通过对原始数据的深入理解与合理处理，为构建高效、稳定的预测模型提供了坚实保障。

# 模型构建与参数优化

本实验采用随机森林回归模型（Random Forest Regressor）对房价进行预测。随机森林是一种集成学习算法，通过构建多个决策树并取其预测结果的平均值来降低过拟合风险、提高模型的泛化能力。该模型特别适合处理具有大量特征并可能存在非线性关系的回归问题。本章节主要介绍基于scikit learn的随机森林回归模型的构建和模型参数优化的过程。

## 模型算法介绍

随机森林回归模型属于袋装（Bagging）集成方法的一种。其基本思想是对训练数据进行有放回的随机采样，生成多个训练子集，并分别训练多个决策树。在预测阶段，将所有树的输出结果进行平均作为最终结果。

随机森林的主要特点如下：

1.  抗过拟合能力强：通过多个树的集成可以显著降低单棵树可能产生的过拟合问题。
2.  非线性建模能力强：可以处理高维数据和非线性特征之间的复杂关系。
3.  特征重要性评估：能够输出各特征的重要性排序，为特征选择与解释提供依据。
4.  对缺失值和异常值鲁棒：本实验中通过数据预处理填充了缺失值以进一步提升稳定性。

同时，随机森林也存在以下缺点：

1.  训练时间较长: 由于随机森林需要构建多棵树并进行集成, 训练时间 通常较长, 尤其在处理大规模数据集时可能会耗时较多。
2.  内存消耗较大: 随机森林需要存储多棵树的信息, 因此对内存的消耗较大。在处理大规模数据集时, 可能需要较大的内存空间。
3.  不适用于高维稀疏数据: 由于随机森林采用了多树集成的方式, 对于高维稀疏数据的处理相对较为困难。在这种情况下, 其他特定的算法可能更加适用。
4.  不适用于序列数据和时间序列数据: 随机森林回归是一种基于树结构的模型, 对于序列数据和时间序列数据的建模较为困难, 可能需要其他特定的方法。

## 参数调整

评分函数设定为自定义的 RMSE（均方根误差），并使用 make_scorer 包装，使其适用于网格搜索。交叉验证先采用5折分割，确保每次训练都覆盖全部样本。模型调参采用 GridSearchCV，先设置了以下参数网格，确定大致范围：

(1) n_estimators：树的数量，取值为50，100，200；

(2) max_depth：树的最大深度，取值为10， 20， 30；

(3) min_samples_split：最小划分样本数，取值为2，5，10；

(4) min_samples_leaf：叶子节点最小样本数，固定为1，2，4；

(5) max_features：最大特征选择方式，取值为'sqrt'， 'log2'， None。

根据结果得出的最佳参数组合:

{'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}

固定叶子节点最小样本数为1，最大特征选择方式为'sqrt'，其他参数重新确定更精细范围：

(1) n_estimators：取值范围为100到500，步长为10；

(2) max_depth：取值范围为10到30，步长为2；

(3) min_samples_split：取值范围为2到9，步长为1；

(4) min_samples_leaf：固定为1；

(5) max_features：固定为 "sqrt"。

从而得出了在五折交叉验证下的最佳参数组合：

{'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 300}

最佳 RMSE (负数表示越小越好)为 -0.1377026382815109，R²达到了0.9810。在Kaggle网站上提交数据，得分为15914.50449（提交结果的评估标准是预测值的对数与实际销售价格的对数之间的均方根误差（RMSE），分数越小越好）。

图4-1 五折交叉验证下最佳参数组合的Kaggle得分

再将模型改为十折交叉验证，同样使用与五折最终相同的网格搜索矩阵，即：(1) n_estimators：取值范围为100到500，步长为10；

(2) max_depth：取值范围为10到30，步长为2；

(3) min_samples_split：取值范围为2到9，步长为1；

(4) min_samples_leaf：固定为1；

(5) max_features：固定为 "sqrt"。

得出在十折交叉验证下的最佳参数组合为：

{'max_depth': 14, 'max_features': 'sqrt', 'min_sampes_leaf': 1, 'min_samples_split': 2, 'n_estimators': 260}

最佳 RMSE 为-0.13606095839513924l，R²为0.9817，在Kaggle网站上提交后，得分为16085.87081。

图4-2 十折交叉验证下最佳参数组合的Kaggle得分

## 测试集预测及提交结果

使用最优模型对测试集数据进行预测，并通过反对数变换还原为真实价格区间，最终输出预测结果为提交文件 submission.csv。其中前20行如下表所示：

表5-1 提交文件submission.csv前20行

|     |     |
| --- | --- |
| Id  | SalePrice |
| 1461 | 122519.2 |
| 1462 | 156260.7 |
| 1463 | 179187.1 |
| 1464 | 184834.9 |
| 1465 | 187727.2 |
| 1466 | 181517.4 |
| 1467 | 173968.8 |
| 1468 | 174448 |
| 1469 | 182683.5 |
| 1470 | 125966.4 |
| 1471 | 200004.1 |
| 1472 | 93746.89 |
| 1473 | 96996.92 |
| 1474 | 151123.1 |
| 1475 | 116064.4 |
| 1476 | 353527.5 |
| 1477 | 246653.4 |
| 1478 | 289666.2 |
| 1479 | 273232.4 |
| 1480 | 453815.2 |

## 本章小结

本章主要介绍了本实验所采用的随机森林回归模型和参数调整的具体过程。由结果可见，在不同的评分体系下，对参数的优劣的体现会有所不同。同时，交叉验证时折数越多不一定就越好。在数据量不够大的情况下，十折交叉验证相比五折每折的数据量更小。同时从R²得分也可看出，折数过多会导致模型过度调参，使得过拟合的发生。

# 数据可视化

## 不同参数的变化对于RMSE的敏感度

通过 GridSearchCV 对随机森林回归器的多个参数组合进行网格搜索与五折交叉验证，模型在不同参数组合下的 RMSE 变化趋势也通过图表清晰展示，揭示了模型对各超参数的敏感程度。图表如下所示：

图5-1 RMSE随最大深度的变化

图5-2 RMSE随最小划分样本数的变化

图5-3 RMSE树的数量的变化

## 特征重要性分析

利用最优模型的 feature_importances_ 属性，对所有特征进行了重要性排序，并可视化前30个特征的贡献度。从结果可见，对房价预测影响最大的变量包括：TotalSF（房屋总使用面积）、OverallQual（整体质量）、GrLivArea（地上居住面积）、YearBuilt（建造年份）、Age（房龄）、TotalBath（总厕所数量）等。其中贡献度最高的前三十个特征如下图所示：

图5-4 贡献度最高的前三十个特征

# 总结与展望

本章对全文的主要工作作了总结，并提出目前研究的缺陷以及需要进一步研究和改进之处。

## 本文总结

本研究以随机森林为基础模型，构建了房价预测系统，并围绕数据预处理、特征工程、模型构建与评估等方面展开了系统性探索。通过对原始房价数据集的清洗、缺失值填充与变量转换，提取出关键影响因子，并以此为基础训练回归模型。在比较不同参数设定和模型结构的基础上，最终得出相对优越的预测结果，验证了基于随机森林的机器学习方法在房价预测任务中的有效性。

### 本文的主要工作

本文主要研究的是基于随机森林回归模型的房价预测，主要工作内容有以下几个方面：

（1）数据清洗与预处理：填补缺失值，对类别变量进行标签编码，对数值型变量执行标准化与归一化处理。

（2）**特征工程**：在原始特征基础上构造诸如房屋总面积（TotalSF）、总卫生间数量（TotalBath）、房龄（Age）等派生变量，以提升模型表达能力。

（3）**建模与调参**：选用随机森林回归器（Random Forest Regressor）作为主要预测模型，并通过网格搜索（GridSearchCV）和交叉验证技术（5折与10折）优化模型超参数。

（4）**性能评估与可视化**：以对数形式的均方根误差（log-RMSE）作为主要评估指标，分析不同参数组合下模型性能表现，并通过图形化方式呈现模型行为及特征重要性。

（5）**模型预测与提交结果**：使用训练好的最佳模型对测试集进行预测，并输出提交格式的房价预测结果文件。

## 局限性

尽管本研究取得了一定成果，仍存在以下几个不足之处：

（1）计算效率较低：模型训练和调参过程中主要依赖网格搜索方法，搜索过程存在计算成本高、耗时长的问题，尤其在参数空间较大时，表现尤为明显。

（2）预测精度有待提升：受限于特征选择方法和模型结构，整体预测性能存在进一步优化空间。

（3）模型所得的结果的R²达到了0.98以上，有过拟合的可能。

## 改进方向

虽然本文采用随机森林模型成功完成了房价预测任务，并实现了较低的对数 RMSE，但在实际建模过程中，仍存在计算效率低，准确度不高等问题，依然有预测准确率与模型鲁棒性的空间。未来的研究可从以下几个方向展开：

（1） 贝叶斯优化与Optuna 的自动超参数调节：本研究初步采用网格搜索进行模型调参，虽然效果良好，但存在搜索效率低、维度灾难等问题。未来可采用基于贝叶斯优化的自动调参工具Optuna，通过构建代理模型（如 TPE）来预测目标函数，从而在较少的搜索次数下高效逼近最优解。

（2） 梯度提升多模型集成：在使用Optuna优化的LightGBM 模型基础上，引入 XGBoost和CatBoost两种性能稳定的梯度提升算法，构建三模型并行预测系统。

（3） 可解释性建模：引入SHAP值分析，为提高预测模型的可解释性，可使用 SHAP 方法评估各特征对模型预测结果的边际贡献。

# 致谢

感谢老师一个学期以来的指导，感谢曾经帮助过我的同学，感谢kaggle网站提供的测试平台和数据集，感谢Scikit-learn提供的模型。

# 参考文献

1.  Housing Prices Competition for Kaggle Learn Users, https://www.kaggle.com/competitions/home-data-for-ml-course/overview
2.  杨艳平, 李荣. 基于机器学习的高维数据分类特征选择\[J\]. 贵州民族大学学报(自然科学版), 2025, 41(1).
3.  Scikit-learn，scikit-learn: machine learning in Python — scikit-learn 1.1.3 documentation
4.  sklearn documentation for RandomForestRegressor, http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
5.  周志华，机器学习，清华大学出版社，2016
6.  Leo Breiman. (2001). “Random Forests.” Machine Learning , 45 (1): 5–32.doi:10.1023/A:10109334043243.

