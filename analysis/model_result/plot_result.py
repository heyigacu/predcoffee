
# import torch
# import matplotlib.pyplot as plt
# # 创建数据
# x = torch.arange(1, 6)
# y_mean = torch.tensor([2, 4, 6, 8, 10], dtype=torch.float32)
# y_std = torch.tensor([0.5, 0.8, 1.2, 1.5, 1.8], dtype=torch.float32)
# # 创建一个6x1的子图布局
# fig, axs = plt.subplots(6, 1, figsize=(8, 16))
# # 循环绘制六个子图
# for i, ax in enumerate(axs):
#     # 绘制每个子图的条形图
#     ax.bar(x, y_mean, yerr=y_std, capsize=4)
#     # 设置每个子图的标题和坐标轴标签
#     ax.set_title(f'Error Bar Chart {i+1}')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
# # 调整子图之间的间距
# plt.tight_layout()
# # 显示图形
# plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

PWD = os.path.abspath(os.path.dirname(__file__))


arr = np.loadtxt(PWD+"/performance.txt")
print(arr.shape)
mean = arr.reshape(10,4,-1).mean(0)
std = arr.reshape(10,4,-1).std(0)

np.savetxt(PWD+'/mean.txt', mean)
np.savetxt(PWD+'/std.txt', std)



coffee_mean = np.loadtxt(PWD+"/coffee_mean.txt")
coffee_std = np.loadtxt(PWD+"/coffee_std.txt")
print(coffee_mean)

mean = np.stack((coffee_mean, mean[0], mean[1], mean[2], mean[3]), axis=0)
std = np.stack((coffee_std, std[0], std[1],std[2], std[3]), axis=0)

with open(PWD+"/table.txt", 'w') as f:
    for i,mean_line in enumerate(mean):
        ls = []
        for j,mean_word in enumerate(mean_line):
            ls.append('{:.2f}'.format(mean_word)+'±'+'{:.2f}'.format(std[i][j]))
        f.write('\t'.join(ls)+'\n')




fig = plt.figure(figsize=(24,4),dpi=600)
cm = np.zeros([2,2])
cm_a = [['',''],['','']]
for i in range(5):
    cm[0][0] = float(mean[i][0])
    cm[0][1] = float(mean[i][1])
    cm[1][0] = float(mean[i][2])
    cm[1][1] = float(mean[i][3])
    cm_a[0][0] = '{:.2f} ± {:.2f}'.format(float(mean[i][0]), float(std[i][0]))
    cm_a[0][1] = '{:.2f} ± {:.2f}'.format(float(mean[i][1]), float(std[i][1]))
    cm_a[1][0] = '{:.2f} ± {:.2f}'.format(float(mean[i][2]), float(std[i][2]))
    cm_a[1][1] = '{:.2f} ± {:.2f}'.format(float(mean[i][3]), float(std[i][3]))
    ax = fig.add_subplot(1, 5, i + 1)
    plt.subplots_adjust(wspace=0.3)
    sns.heatmap(cm, ax=ax, annot=cm_a, cbar=False, annot_kws={"size": 14}, fmt='',cmap="Blues", xticklabels = ['non-coffee', 'coffee', ], yticklabels = ['non-coffee', 'coffee'])
    # plt.ylabel('True label', fontsize=16)
    # plt.xlabel('Predicted label', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
plt.savefig(PWD+'/cm.png', bbox_inches='tight') 




labels = ['Precise','Accuracy','AP','F1','MCC','AUC']
colors = ['#2878B5', '#F8AC8C', '#BEB8DC', '#D76364']
X = ['KPGT','MLP','RF','SVM','MPNN']
mean = mean[:,-6:]
std = std[:,-6:]
fig = plt.figure(figsize=(8,8),dpi=600)
for i in range(6):
    y = mean[:,i]
    # yerr = std[:,i]
    ax = fig.add_subplot(3, 2, i + 1)
    ax.bar(X, y, yerr=None, color=colors,alpha=0.5, capsize=5)
    ax.set_xticks(X)
    ax.set_xticklabels(X)
    ax.set_title('{}'.format(labels[i]))
plt.tight_layout()
plt.savefig(PWD+'/pm.png', bbox_inches='tight') 

