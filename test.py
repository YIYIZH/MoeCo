import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 假设这是你的数据，名为data1和data2（作为列表或NumPy数组）
# IVT
# data2 = [np.nan, 0.00934234, 0.1, 0.01872989, 0.38670039, np.nan,
#  0.42350491, 0.45683599, 0.00167183, 0.03319888, 0.22008648, np.nan,
#  0.90152894, 0.73746791, np.nan, np.nan, 0.38166516, 0.89953998,
#  0.16578521, 0.70413422, 0.61342412, 0.4013539, 1.0, np.nan,
#  np.nan, 0.12910397, 0.92255952, 0.30986529, 0.43958641, 0.84711478,
#  0.83632431, np.nan, np.nan, np.nan, 0.07710878, np.nan,
#  0.13860594, np.nan, 0.00363975, 0.46386082, np.nan, 0.00352065,
#  np.nan, np.nan, 0.42825041, 0.13345162, np.nan, np.nan,
#  np.nan, np.nan, np.nan, 0.01141415, 0.01160799, 0.35386702,
#  np.nan, np.nan, np.nan, 0.45447936, 0.50102431, 0.3926738,
#  0.80165875, 0.76339131, 0.55270216, 0.39942039, 0.03818248, np.nan,
#  0.26721655, np.nan, 0.55886126, 0.72624289, 1.0, 0.20906798,
#  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
#  0.64416296, 0.68869536, np.nan, 0.9537037, 0.74762353, np.nan,
#  np.nan, np.nan, np.nan, np.nan, 0.50304072, 0.0378537,
#  0.23925131, np.nan, 0.20435251, np.nan, 0.3229727, 0.23125729,
#  0.36118698, 0.20449552, 0.35316246, 0.17377069]

# data1 = [np.nan, 0.01759956, 0.05555556, 0.02677719, 0.22439385, np.nan, 0.07812339, 0.47156467, 0.00763736, 0.02666343, 0.01150223, np.nan,0.91709187, 0.49178286, np.nan, np.nan, 0.26303627, 0.92099583,
#  0.12686869, 0.72016208, 0.70305317, 0.31675648, 0.97833619, np.nan,
#  0.78358443, np.nan, np.nan, np.nan, 0.03576596, np.nan,
#  0.20659499, np.nan, 0.0029499, 0.05747759, np.nan, 0.09450847,
#  np.nan, np.nan, 0.0905218, 0.0685207, np.nan, np.nan,
#  np.nan, np.nan, 0.02773698, 0.01161177, 0.19087763,
#  np.nan, np.nan, np.nan, 0.41597747, 0.56498782, 0.2932609,
#  0.87666494, 0.75912516, 0.57871277, 0.08746797, 0.06233406, np.nan,
#  0.06539851, np.nan, 0.7072824, 0.691877, 0.40745887, 0.02848049,
#  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
#  0.62831351, 0.69152343, np.nan, 0.47817688, 0.69483093, np.nan,
#  np.nan, np.nan, np.nan, np.nan, 0.29463066, 0.01639448,
#  0.30197157, np.nan, 0.184542, np.nan]
# I
# data2 = [0.95526085, 0.94864992, 0.98159141, 0.75968901, 0.87340342, 0.96126993]
# data1 = [0.96602274, 0.93481697, 0.98330015, 0.73324069, 0.87442816, 0.93185996]

# V
# data2 = [0.77321486, 0.94361406, 0.9407941, 0.79781708, 0.80596763, 0.73980349,
#  0.74762353, 0.37184624, 0.73746791, 0.3686172]
# data1 = [0.7527431,  0.95949106, 0.95169929, 0.82797173, 0.80439639, 0.72419409,
#  0.69483093, 0.34115915, 0.49178286, 0.32691167]

# T
data2 = [0.94436426, 0.22980101, 0.54479673, 0.33926226, 0.46462955, np.nan,
 0.74762353, 0.3737137, 0.72038841, 0.81338725, 0.75246509, 0.53425172,
 0.16686095, 0.90152894, 0.3686172]
data1 = [0.96207385, 0.18125496, 0.56216422, 0.32444431, 0.30252246, np.nan,
 0.69483093, 0.31191592, 0.746402,   0.15423764, 0.78596217, 0.47574226,
 0.12831436, 0.91663772, 0.32691167]

# 将数据转换为DataFrame格式，便于Seaborn处理
# 这里假设两组数据代表两个不同的类别或条件
df1 = pd.DataFrame({'value': data1, 'group': 'TERL-T'})
df2 = pd.DataFrame({'value': data2, 'group': 'MoeCo-T'})
df = pd.concat([df1, df2], ignore_index=True)

# 清除NaN值 - 根据你的分析目标选择是否删除
# 箱线图通常能自动处理NaN，但散点图需要具体坐标
df_clean = df.dropna(subset=['value'])

# 创建图形
plt.figure(figsize=(10, 6)) # 设置图形大小

plt.rcParams['font.size'] = 28
# 设置全局字体粗细
# plt.rcParams['font.weight'] = 'bold'

# 首先绘制箱线图
# sns.boxplot(x='group', y='value', data=df_clean, color='lightgray', width=0.4)
sns.boxplot(x='group', y='value', data=df_clean, width=0.4,
            boxprops=dict(facecolor='lightyellow', edgecolor='black'), # coral，yellow，green， blue，
            medianprops=dict(color='blue', linewidth=8),
            showfliers=False)

# 然后叠加散点图
# 使用stripplot或swarmplot来显示数据点分布
# stripplot: 点可能会重叠
# swarmplot: 点会避免重叠，但计算量更大，对于大数据集可能较慢
sns.stripplot(x='group', y='value', data=df_clean, color='red', alpha=0.5, jitter=True, size=10, edgecolor='gray')

# 添加标题和标签
plt.title('')
plt.xlabel('')
plt.ylabel(r'AP$_{T}$')

plt.grid(True, axis='y', linestyle='--', alpha=0.7, color='grey')
plt.tight_layout()

# 显示图形
# plt.show()
plt.savefig('box_t.pdf', format='pdf')
