import math

# 输入数据矩阵，2行6列
DM = [[60, 70, 50, 140, 2, 1],
      [20, 30, 80, 150, 3, 0]]

# 初始化归一化矩阵
gui_yi_DM = [[0] * 6 for _ in range(2)]
print(gui_yi_DM)

# 计算归一化矩阵
for i in range(6):
    temp_sum = math.sqrt(DM[0][i] ** 2 + DM[1][i] ** 2)  # 计算每一列的平方和的平方根
    gui_yi_DM[0][i] = DM[0][i] / temp_sum  # 第一行的归一化
    gui_yi_DM[1][i] = DM[1][i] / temp_sum  # 第二行的归一化

print(f"归一化DM：{gui_yi_DM}")

# 计算加权归一化决策矩阵（这里的权重是均匀的，每个权重为1/6）
WNDM = [[val / 6 for val in row] for row in gui_yi_DM]

print(f"WNDM：{WNDM}")

# 提取加权归一化矩阵的两行
WNDM1 = WNDM[0]
WNDM2 = WNDM[1]

# 计算正理想解（A+）和负理想解（A-）
A_jia = []
A_jian = []

for i in range(3):  # 假设前3列是效益型指标
    da = max(WNDM1[i], WNDM2[i])
    xiao = min(WNDM1[i], WNDM2[i])
    A_jia.append(da)
    A_jian.append(xiao)

for i in range(3, 5):  # 假设第4, 5列是成本型指标
    da = min(WNDM1[i], WNDM2[i])
    xiao = max(WNDM1[i], WNDM2[i])
    A_jia.append(da)
    A_jian.append(xiao)

# 处理最后一个指标（假设也是效益型）
da = max(WNDM1[5], WNDM2[5])
xiao = min(WNDM1[5], WNDM2[5])
A_jia.append(da)
A_jian.append(xiao)

print(f"A+：{A_jia}")
print(f"A-：{A_jian}")

# 计算每个方案与正理想解和负理想解的距离
SM_jia = []
SM_jian = []

SM1_jia = sum((WNDM1[i] - A_jia[i]) ** 2 for i in range(6))
SM2_jia = sum((WNDM2[i] - A_jia[i]) ** 2 for i in range(6))

SM1_jian = sum((WNDM1[i] - A_jian[i]) ** 2 for i in range(6))
SM2_jian = sum((WNDM2[i] - A_jian[i]) ** 2 for i in range(6))

SM_jia.append(math.sqrt(SM1_jia))
SM_jia.append(math.sqrt(SM2_jia))
SM_jian.append(math.sqrt(SM1_jian))
SM_jian.append(math.sqrt(SM2_jian))

print(f"SM+：{SM_jia}")
print(f"SM-：{SM_jian}")

# 计算相对接近度RC
RC = []
RC1 = SM_jian[0] / (SM_jia[0] + SM_jian[0])
RC2 = SM_jian[1] / (SM_jia[1] + SM_jian[1])
RC.append(RC1)
RC.append(RC2)

print(f"RC：{RC}")
