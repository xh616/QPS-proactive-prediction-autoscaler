import math

# 输入数据矩阵，4行4列
DM = [
    [60, 70, 50, 140],
    [20, 30, 80, 150],
    [40, 60, 30, 120],
    [30, 20, 100, 110]
]
# DM = [[60, 70, 50, 140, 2, 1],
#       [20, 30, 80, 150, 3, 0]]
# 初始化归一化矩阵
rows, cols = len(DM), len(DM[0])  # 获取行数和列数
gui_yi_DM = [[0] * cols for _ in range(rows)]

# 计算归一化矩阵
for j in range(cols):
    temp_sum = math.sqrt(sum(DM[i][j] ** 2 for i in range(rows)))  # 计算每一列的平方和的平方根
    for i in range(rows):
        gui_yi_DM[i][j] = DM[i][j] / temp_sum  # 每个元素归一化

print(f"归一化DM：{gui_yi_DM}")

# 计算加权归一化决策矩阵（这里的权重是均匀的，每个权重为1/4）
WNDM = [[val / cols for val in row] for row in gui_yi_DM]

print(f"WNDM：{WNDM}")

# 计算正理想解（A+）和负理想解（A-）
A_jia = []
A_jian = []

for j in range(3):  # 处理效益型指标
    da = max(WNDM[i][j] for i in range(rows))
    xiao = min(WNDM[i][j] for i in range(rows))
    A_jia.append(da)
    A_jian.append(xiao)

for j in range(4):  # 处理成本型指标
    da = min(WNDM[i][j] for i in range(rows))
    xiao = max(WNDM[i][j] for i in range(rows))
    A_jia.append(da)
    A_jian.append(xiao)

print(f"A+：{A_jia}")
print(f"A-：{A_jian}")

# 计算每个方案与正理想解和负理想解的距离
SM_jia = []
SM_jian = []

for i in range(rows):
    SM_i_jia = math.sqrt(sum((WNDM[i][j] - A_jia[j]) ** 2 for j in range(cols)))
    SM_i_jian = math.sqrt(sum((WNDM[i][j] - A_jian[j]) ** 2 for j in range(cols)))
    SM_jia.append(SM_i_jia)
    SM_jian.append(SM_i_jian)

print(f"SM+：{SM_jia}")
print(f"SM-：{SM_jian}")

# 计算相对接近度RC
RC = [SM_jian[i] / (SM_jia[i] + SM_jian[i]) for i in range(rows)]

print(f"RC：{RC}")

# 找出RC列表中最大的值的索引
max_index = RC.index(max(RC))

print(f"最大的RC值索引：{max_index}")
