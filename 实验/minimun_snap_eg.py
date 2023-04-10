# -*- coding: utf-8 -*-
#%% 生成路径点
import numpy as np
import matplotlib.pyplot as plt

path = [[1, 3], [3, 5], [4, 2], [2.5, 1.2], [2, -2.5]]
path = np.array(path)
#plt.plot(path[:, 0], path[:, 1])
plt.plot(path[:, 0], path[:, 1], "*")
plt.show()

#%% 一维轨迹优化
x = path[:, 0]
# 分配时间
deltaT = 2 # 每一段2秒
T = np.linspace(0, deltaT * (len(x) - 1), len(x))

########### 目标函数 ###########
######## 1/2xTQx + qTx ########

K = 3                   # jerk为3阶导数，取K=3
n_order = 2 * K - 1     # 多项式阶数
M = len(x) - 1          # 轨迹的段数
N = M * (n_order + 1)   # 矩阵Q的维数

def getQk(T_down, T_up):
    Q = np.zeros((6, 6))
    Q[3][4] =  72 * (T_up**2 - T_down**2)
    Q[3][5] = 120 * (T_up**3 - T_down**3)
    Q[4][5] = 360 * (T_up**4 - T_down**4)
    Q = Q + Q.T # Q为对称矩阵
    Q[3][3] =  36 * (T_up**1 - T_down**1)
    Q[4][4] = 192 * (T_up**3 - T_down**3)
    Q[5][5] = 720 * (T_up**5 - T_down**5)
    return Q
    
Q = np.zeros((N, N))
for k in range(1, M + 1):
    Qk = getQk(T[k - 1], T[k])
    Q[(6 * (k - 1)) : (6 * k), (6 * (k - 1)) : (6 * k)] = Qk

Q = Q * 2 # 因为标准目标函数为： 1/2xTQx + qTx，所以要乘2

########### 约束 ###########
# 1.导数约束 Derivative Constraint
A0 = np.zeros((2 * K + M - 1, N)) 
b0 = np.zeros(len(A0))

# 添加首末状态约束(包括位置、速度、加速度)
for k in range(K): 
    for i in range(k, 6):
        c = 1
        for j in range(k):
            c *= (i - j)
        A0[0 + k * 2][i]                = c * T[0]**(i - k)
        A0[1 + k * 2][(M - 1) * 6 + i]  = c * T[M]**(i - k)
b0[0] = x[0]
b0[1] = x[M]
# 添加每段轨迹的初始位置约束
for m in range(1, M):
    for i in range(6):
        A0[6 + m - 1][m * 6 + i] = T[m]**i
    b0[6 + m - 1] = x[m]

# 2.连续性约束 Continuity Constraint
A1 = np.zeros(((M - 1) * 3, N))
b1 = np.zeros(len(A1))
for m in range(M - 1):
    for k in range(3): # 最多两阶导数相等    
        for i in range(k, 6):
            c = 1
            for j in range(k):
                c *= (i - j)
            index = m * 3 + k
            A1[index][m * 6 + i] = c * T[m + 1]**(i - k)
            A1[index][(m + 1)* 6 + i] = -c * T[m + 1]**(i - k)
A = np.vstack((A0, A1))
b = np.hstack((b0, b1))
#%% 解二次规划问题
from cvxopt import matrix, solvers
# 目标函数
Q = matrix(Q)
q = matrix(np.zeros(N))
# 等式约束: Ax = b
A = matrix(A)
b = matrix(b)
result = solvers.qp(Q, q, A=A, b=b)
p_coff = np.asarray(result['x']).flatten()

#%% 可视化x轴优化结果
Pos = []
Vel = []
Acc = []
for k in range(M):
    t = np.linspace(T[k], T[k + 1], 100)
    t_pos = np.vstack((t**0, t**1, t**2, t**3, t**4, t**5))
    t_vel = np.vstack((t*0, t**0, 2 * t**1, 3 * t**2, 4 * t**3, 5 * t**4))
    t_acc = np.vstack((t*0, t*0, 2 * t**0, 3 * 2 * t**1, 4 * 3 * t**2, 5 * 4 * t**3))
    coef = p_coff[k * 6 : (k + 1) * 6]
    coef = np.reshape(coef, (1, 6))
    pos = coef.dot(t_pos)
    vel = coef.dot(t_vel)
    acc = coef.dot(t_acc)
    Pos.append([t, pos[0]])
    Vel.append([t, vel[0]])
    Acc.append([t, acc[0]])

Pos = np.array(Pos)
Vel = np.array(Vel)
Acc = np.array(Acc)
plt.subplot(3, 1, 1)
plt.plot(Pos[:, 0, :].T, Pos[:, 1, :].T)
#plt.title("position")
plt.xlabel("time(s)")
plt.ylabel("position(m)")
plt.subplot(3, 1, 2)
plt.plot(Vel[:, 0, :].T, Vel[:, 1, :].T)
# plt.title("velocity")
plt.xlabel("time(s)")
plt.ylabel("velocity(m/s)")
plt.subplot(3, 1, 3)
plt.plot(Acc[:, 0, :].T, Acc[:, 1, :].T)
# plt.title("accel")
plt.xlabel("time(s)")
plt.ylabel("accel(m/s^2)")
plt.show()