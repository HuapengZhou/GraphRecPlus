import matplotlib.pyplot as plt

# 设置数据
labels = ['Training', 'Testing']
graphrec_plus_times = [3.86, 10.42]
graphrec_times = [4.66, 12.95]

x = range(len(labels))  # the label locations

# 设置图形大小
plt.figure(figsize=(10, 6))

# 创建柱状图
plt.bar(x, graphrec_plus_times, width=0.4, label='GraphRec + Resnet', color='b', align='center')
plt.bar(x, graphrec_times, width=0.4, label='GraphRec', color='r', align='edge')

# 添加标签、标题等
plt.xlabel('Process')
plt.ylabel('Time (s)')
plt.title('Training and Testing Times')
plt.xticks(x, labels)
plt.legend()

# 在每个条形上显示数值
for i in range(len(x)):
    plt.text(x[i], graphrec_plus_times[i], '%.2f' % graphrec_plus_times[i], ha='center', va= 'bottom',fontsize=11, color='b')
    plt.text(x[i], graphrec_times[i], '%.2f' % graphrec_times[i], ha='left', va= 'bottom',fontsize=11, color='r')

# 显示图形
plt.tight_layout()
plt.show()
