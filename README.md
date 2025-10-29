# 顾客细分聚类项目（Project 2: Clustering）

本项目基于附件“project_2_clustering.pdf”的要求与流程，使用商场顾客数据集进行无监督聚类，比较 K-Means、层次聚类（Hierarchical Clustering）与 DBSCAN 等方法，完成特征工程、参数选择、可视化与结果解读，产出可复现的分析指南。

> 数据文件：`Mall_Customers.csv`


## 项目目标

- 对顾客进行无监督细分，发现具有业务意义的群体（如高消费高潜力、价格敏感等）。
- 对比多种聚类算法的适用性与表现，给出参数选择的依据与评价指标。
- 输出直观的聚类可视化和业务可解释的画像描述，辅助营销与会员运营。


## 数据集说明

数据源为经典的商场顾客数据（常见字段如下，具体以 `Mall_Customers.csv` 为准）：

- `CustomerID`：顾客编号（标识，不参与建模）
- `Genre`：性别（分类特征）
- `Age`：年龄（数值特征）
- `Annual Income (k$)`：年收入（千美元，数值特征）
- `Spending Score (1-100)`：消费得分（1-100，数值特征）

常用特征工程策略：

- 删除或保留 `CustomerID` 作为索引；
- 将 `Genre` 编码为 0/1（或独热编码）；
- 数值特征标准化（StandardScaler）或缩放到 [0,1]（MinMaxScaler），保证算法尺度一致；
- 根据 PDF 要求与探索性分析（EDA）结果，选择用于聚类的最相关特征子集（如 `Age`、`Annual Income (k$)`、`Spending Score (1-100)`）。


## 方法与指标（仅作示例参考）

### 1) K-Means

- 通过肘部法（Elbow）选择聚类数 K：绘制 K 与 SSE（簇内平方误差和）的关系曲线，拐点对应的 K 为候选。
- SSE 定义为：$J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \lVert x_i - \mu_k \rVert^2$。
- 使用轮廓系数（Silhouette Coefficient）进一步评估不同 K 的聚类质量：
  $s(i) = \dfrac{b(i) - a(i)}{\max\{a(i), b(i)\}}$，其中 $a(i)$ 为样本到同簇内其余样本的平均距离，$b(i)$ 为样本到最近其他簇的平均距离。
- 实践要点：设置 `n_init`、`random_state` 提升稳定性；标准化特征以减小尺度偏置。

### 2) 层次聚类（Hierarchical）

- 距离度量与链接方式常用：`ward`（默认，最小化方差）或 `average`、`complete`。
- 通过树状图（dendrogram）观察层次结构，依据距离阈值或簇数切分。
- 适合小中等规模数据，能揭示簇间层级关系。

### 3) DBSCAN

- 基于密度的聚类，能发现任意形状的簇并识别噪声点。
- 核心参数：`eps`（半径）与 `min_samples`（最小样本数）。可使用 k-distance 曲线选择 `eps` 拐点。
- 对尺度敏感，建议先标准化；当簇密度差异较大时可能需要分段调参或改用 HDBSCAN。

### 4) 评价与可视化

- 内部指标：平均 Silhouette、Davies–Bouldin 指数（DBI）等；
- 外部指标：若有标签可用 ARI、NMI，但本项目为无监督通常不可用；
- 可视化：二维特征对散点图、PCA/t-SNE/U-MAP 降维后着色、热力图展示特征均值等。


## 实验流程建议（与 PDF 对齐）

1. 数据读取与检查：缺失值、异常值、唯一值分布；
2. 探索性分析（EDA）：分布、箱线图、相关性热力图；
3. 特征工程：类别编码、数值标准化、特征选择；
4. K-Means：
	- 画 Elbow 曲线（SSE vs. K，K∈[2,10] 等范围）；
	- 画 Silhouette 曲线（平均轮廓系数 vs. K），综合选择最优 K（例如常见数据上可能出现 K≈5，仅作示例，实际以你的拐点与指标为准）；
	- 训练模型、得到簇标签与中心，制作聚类散点图与特征均值雷达图/柱状图；
5. 层次聚类：
	- 选择链接方式，绘制树状图并确定簇数；
	- 与 K-Means 结果对比（簇内紧致度、业务可解释性）；
6. DBSCAN：
	- 基于 k-distance 曲线调 `eps`，尝试多组 `min_samples`；
	- 分析噪声点比例与簇形态，若过多噪声/簇碎片化，适度调参或回退其他方法；
7. 结果汇总与业务画像：
	- 输出每个簇的样本数、关键特征均值/分位数；
	- 给出营销策略建议（如高价值维护、潜力提升、价格敏感优惠券等）；
8. 结论与局限：
	- 哪种方法在本数据更稳健/易解释；
	- 对超参数、尺度、异常值的敏感度；
	- 下一步可尝试的改进（更多行为特征、时序特征或 HDBSCAN/高斯混合等）。


## 复现指南（建议环境）

依赖（示例）：

- Python 3.9+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- scipy

快速流程（仅作示例参考）：

1) 将 `Mall_Customers.csv` 放在仓库根目录（已包含）。

2) 按“实验流程建议”在你偏好的环境中执行（脚本或 Notebook）。若使用 Notebook，可参考以下基本骨架：

```python
# 1) 读取与预处理
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Mall_Customers.csv')
# 视具体列名而定：
# df = df.rename(columns={'Annual Income (k$)':'Income', 'Spending Score (1-100)':'Score'})
# 类别编码、缺失处理...

X = df[[
	 'Age',
	 'Annual Income (k$)',
	 'Spending Score (1-100)'
]].copy()
X = StandardScaler().fit_transform(X)

# 2) KMeans + 评价
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sse, sil = [], []
for k in range(2, 11):
	 km = KMeans(n_clusters=k, n_init=10, random_state=42)
	 labels = km.fit_predict(X)
	 sse.append(km.inertia_)
	 sil.append(silhouette_score(X, labels))

# 3) 选择K并训练、可视化（略）
```

如需产出图表，可用 matplotlib/seaborn 绘制 Elbow、Silhouette、聚类散点图与热力图。


## 文件结构

```
.
├── Mall_Customers.csv        # 数据集
└── README.md                 # 项目说明（当前文件）
```


## 结果与讨论（模板）（仅作示例参考）

请在完成实验后填充本节关键发现：

- 最优方法与参数：例如 K-Means（K=...）、层次聚类（链接=...）、DBSCAN（eps=...，min_samples=...）；
- 各簇规模与画像：年龄/收入/消费得分的均值或分位数，业务含义；
- 指标对比：平均 Silhouette、可视化分离度、稳定性；
- 商业建议：针对不同簇的营销与运营策略。


## 参考

- sklearn: Clustering — KMeans, AgglomerativeClustering, DBSCAN
- 轮廓系数（Silhouette）、Davies–Bouldin 指数（DBI）
- 经典 Mall Customers 数据集与相关公开案例


## 许可

若无特别声明，默认以学术学习用途为主。根据你的需求补充 LICENSE。

