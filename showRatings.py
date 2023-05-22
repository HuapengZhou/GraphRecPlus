import scipy.io
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

# Load the .mat file.
mat2 = scipy.io.loadmat('data/Ciao/rating.mat')
# Print the keys of the mat dictionary to see the variables stored in the .mat file.

print(mat2.keys())
ratings = mat2['rating']
# 打乱数据
np.random.shuffle(ratings)

# 取前1000条数据
num_samples = 10000
ratings = ratings[:num_samples, :]

# 获取所需的列数据
userid = ratings[:, 0]
productid = ratings[:, 1]
categoryid = ratings[:, 2]
rating = ratings[:, 3]
helpfulness = ratings[:, 4]

# 以下是可视化的代码，与之前的代码相同
# 取前10个用户的评分数据
unique_users, user_counts = np.unique(userid, return_counts=True)
top_users = unique_users[:10]

plt.figure(figsize=(12, 8))
for i, user in enumerate(top_users):
    user_ratings = rating[userid == user]
    plt.plot(user_ratings, label='User {}'.format(user))

plt.xlabel('Product Index')
plt.ylabel('Rating')
plt.title('Product Ratings by Top 10 Users')
plt.legend()
plt.show()
# 使用条形图可视化平均评分
unique_products, product_counts = np.unique(productid, return_counts=True)
average_ratings = np.zeros_like(unique_products, dtype=float)
for i, product in enumerate(unique_products):
    product_ratings = rating[productid == product]
    average_ratings[i] = np.mean(product_ratings)

plt.figure(figsize=(10, 6))
plt.bar(unique_products, average_ratings)
plt.xlabel('Product ID')
plt.ylabel('Average Rating')
plt.title('Average Rating per Product')
plt.show()

# 使用散点图可视化评分分布
plt.figure(figsize=(10, 6))
plt.scatter(productid, rating, c=userid)
plt.colorbar(label='User ID')
plt.xlabel('Product ID')
plt.ylabel('Rating')
plt.title('Rating Distribution')
plt.show()

# 使用热力图可视化评分情况
unique_products = np.unique(productid)
unique_categories = np.unique(categoryid)
rating_matrix = np.zeros((len(unique_categories), len(unique_products)), dtype=float)
for i, category in enumerate(unique_categories):
    for j, product in enumerate(unique_products):
        product_ratings = rating[(productid == product) & (categoryid == category)]
        if len(product_ratings) > 0:
            rating_matrix[i, j] = np.mean(product_ratings)

plt.figure(figsize=(12, 8))
plt.imshow(rating_matrix, cmap='hot', aspect='auto')
plt.colorbar(label='Rating')
plt.xticks(np.arange(len(unique_products)), unique_products, rotation='vertical')
plt.yticks(np.arange(len(unique_categories)), unique_categories)
plt.xlabel('Product ID')
plt.ylabel('Category ID')
plt.title('Rating Heatmap')
plt.show()

