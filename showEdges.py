import scipy.io
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

# Load the .mat file.
mat1 = scipy.io.loadmat('data/Ciao/trustnetwork.mat')
mat2 = scipy.io.loadmat('data/Ciao/rating.mat')
# Print the keys of the mat dictionary to see the variables stored in the .mat file.
print(mat1.keys())
print(mat2.keys())
# To print a part of the file, you can do:
# Let's assume that the name of a variable in the .mat file is 'variable_name'.

# print(len(variable_data))  # prints the first five rows of 'variable_name'
# variable_data2 = mat2['rating']
# print(variable_data2[:50])  # prints the first five rows of 'variable_name'

edges = mat1['trustnetwork']
print(type(edges))
# 打乱边的顺序
np.random.shuffle(edges)

# 取前10000个边（如果 edges 中的边数大于10000）
edges = edges[:1000] if edges.shape[0] > 10000 else edges
print(edges)
# 创建一个空的无向图
G = nx.Graph()

# 将边添加到图中，注意这里我们需要将 ndarray 转换为 tuple 列表
G.add_edges_from([tuple(edge) for edge in edges])

# 使用shell_layout作为布局
pos = nx.shell_layout(G)


# 使用不同的节点颜色、大小和边颜色
node_color = [G.degree(v) for v in G]
node_size = [0.0005 * nx.degree(G, v) for v in G]
# 生成随机颜色列表
edge_color = [random.random() for _ in G.edges()]

# 绘制图
nx.draw(G, pos, node_color=node_color, node_size=node_size, edge_color=edge_color, with_labels=False, linewidths=0.1, width=0.1)

# 保存图形到指定文件夹
plt.savefig("Result/images/graph.png")

# 显示图形
plt.show()