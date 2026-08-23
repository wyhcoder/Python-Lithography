import numpy as np
import matplotlib.pyplot as plt

# 读取第一个 dat 文件（假设有两列：x, y）
data1 = np.loadtxt('mprofile_20260311112313.dat')  # 可根据分隔符指定 delimiter，如 delimiter=','
x1, y1 = data1[:, 0], data1[:, 1]

# 读取第二个 dat 文件
data2 = np.loadtxt('mprofile_20260311113628.dat')
x2, y2 = data2[:, 0], data2[:, 1]

# 在同一张图上绘制
plt.plot(x1, y1, label='File 1', marker='o', linestyle='-')
plt.plot(x2, y2, label='File 2', marker='s', linestyle='--')

# 添加图例、标签等
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Two .dat files on the same plot')
plt.legend()
plt.grid(True)
plt.show()