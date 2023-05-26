import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import pickle
import numpy as np
import time
import random
from collections import defaultdict
from UV_Encoders import UV_Encoder
from UV_Aggregators import UV_Aggregator
from Social_Encoders import Social_Encoder
from Social_Aggregators import Social_Aggregator
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import datetime
import argparse
import os
import matplotlib.pyplot as plt
# TODO: copilot
import tqdm

# check for some error
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

"""
GraphRec: Graph Neural Networks for Social Recommendation. 
Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin. 
In Proceedings of the 28th International Conference on World Wide Web (WWW), 2019. Preprint[https://arxiv.org/abs/1902.07243]

If you use this code, please cite our paper:
```
@inproceedings{fan2019graph,
  title={Graph Neural Networks for Social Recommendation},
  author={Fan, Wenqi and Ma, Yao and Li, Qing and He, Yuan and Zhao, Eric and Tang, Jiliang and Yin, Dawei},
  booktitle={WWW},
  year={2019}
}
```

"""


# class GraphRec(nn.Module):
#
#     def __init__(self, enc_u, enc_v_history, r2e):
#         super(GraphRec, self).__init__()
#         self.enc_u = enc_u
#         self.enc_v_history = enc_v_history
#         self.embed_dim = enc_u.embed_dim
#
#         self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
#         self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
#         self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
#         self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
#         self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
#         self.w_uv2 = nn.Linear(self.embed_dim, 16)
#         self.w_uv3 = nn.Linear(16, 1)
#         self.r2e = r2e
#         self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
#         self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
#         self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
#         self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
#         self.criterion = nn.MSELoss()
#
#     def forward(self, nodes_u, nodes_v):
#         embeds_u = self.enc_u(nodes_u)
#         embeds_v = self.enc_v_history(nodes_v)
#
#         x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
#         x_u = F.dropout(x_u, training=self.training)
#         x_u = self.w_ur2(x_u)
#         x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
#         x_v = F.dropout(x_v, training=self.training)
#         x_v = self.w_vr2(x_v)
#
#         x_uv = torch.cat((x_u, x_v), 1)
#         x = F.relu(self.bn3(self.w_uv1(x_uv)))
#         x = F.dropout(x, training=self.training)
#         x = F.relu(self.bn4(self.w_uv2(x)))
#         x = F.dropout(x, training=self.training)
#         scores = self.w_uv3(x)
#         return scores.squeeze()
#
#     def loss(self, nodes_u, nodes_v, labels_list):
#         scores = self.forward(nodes_u, nodes_v)
#         return self.criterion(scores, labels_list)

class GraphRec(nn.Module):

    def __init__(self, enc_u, enc_v_history, r2e):
        super(GraphRec, self).__init__()
        self.enc_u = enc_u
        self.enc_v_history = enc_v_history
        self.embed_dim = enc_u.embed_dim

        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim * 2)
        self.w_uv2 = nn.Linear(self.embed_dim * 2, 16)
        self.res_uv = nn.Linear(self.embed_dim * 2, 16)  # 新增的线性层，用于调整 x_uv 的尺寸
        self.w_uv3 = nn.Linear(16, 1)
        self.r2e = r2e
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim * 2, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.criterion = nn.MSELoss()

    def forward(self, nodes_u, nodes_v):
        embeds_u = self.enc_u(nodes_u)
        embeds_v = self.enc_v_history(nodes_v)

        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u) + embeds_u

        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v) + embeds_v

        x_uv = torch.cat((x_u, x_v), 1) # TODO: concat changes +
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x))) + self.res_uv(x_uv)  # 调整 x_uv 的尺寸后执行残差连接
        x = F.dropout(x, training=self.training)
        scores = self.w_uv3(x)
        return scores.squeeze()

    def loss(self, nodes_u, nodes_v, labels_list):
        scores = self.forward(nodes_u, nodes_v)
        return self.criterion(scores, labels_list)




def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae):
    model.train()
    running_loss = 0.0
    # for i, data in enumerate(train_loader, 0):
    # TODO:  shrink the train_loader
    loss_values = []
    for i, data in enumerate(tqdm.tqdm(train_loader)):
        batch_nodes_u, batch_nodes_v, labels_list = data
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        # loss_values.append(loss.item())
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                epoch, i, running_loss / 100, best_rmse, best_mae))
            running_loss = 0.0
    # loss_values = np.array(loss_values)

    # 绘制损失曲线
    # plt.plot(loss_values)
    # plt.xlabel('Training Steps')
    # plt.ylabel('Loss')
    # plt.title('Training Loss Curve')
    # plt.show()
    return 0


def test(model, device, test_loader):
    model.eval()
    tmp_pred = []
    target = []
    with torch.no_grad():
        # for i, data in tqdm.tqdm(enumerate(train_loader)):
        for test_u, test_v, tmp_target in tqdm.tqdm(test_loader):
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v)
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    return expected_rmse, mae


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
        print("I am using cuda") # my own add
    device = torch.device("cuda" if use_cuda else "cpu")

    embed_dim = args.embed_dim
    dir_data = './data/Ciao'

    path_data = dir_data + ".pickle"
    data_file = open(path_data, 'rb')
    # history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, social_adj_lists, ratings_list = pickle.load(
    #     data_file)
    with open(path_data, 'rb') as data_file:
        data = pickle.load(data_file)

    # 从data中获取需要的数据
    # TODO: Sample the test
    history_u_lists = data[0]
    history_ur_lists = data[1]
    history_v_lists = data[2]
    history_vr_lists = data[3]
    train_u = data[4]
    train_v = data[5]
    train_r = data[6]
    test_u = data[7]
    test_v = data[8]
    test_r = data[9]
    social_adj_lists = data[10]
    ratings_list = data[11]
    # print(train_u[0:5])
    # print(train_v[0:5])
    """
    ## toy dataset 
    history_u_lists, history_ur_lists:  user's purchased history (item set in training set), and his/her rating score (dict)
    history_v_lists, history_vr_lists:  user set (in training set) who have interacted with the item, and rating score (dict)
    
    train_u, train_v, train_r: training_set (user, item, rating)
    test_u, test_v, test_r: testing set (user, item, rating)
    
    # please add the validation set
    
    social_adj_lists: user's connected neighborhoods
    ratings_list: rating value from 0.5 to 4.0 (8 opinion embeddings)
    """

    # trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
    #                                           torch.FloatTensor(train_r))
    # testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
    #                                          torch.FloatTensor(test_r))
    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)
    from torch.utils.data import TensorDataset, DataLoader, random_split

    # 创建数据集
    trainset = TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v), torch.FloatTensor(train_r))
    testset = TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v), torch.FloatTensor(test_r))

    # 设定要使用的数据的比例
    train_frac = 0.01  # 使用一半的训练数据
    test_frac = 0.1  # 使用一半的测试数据

    # 计算新的训练数据和测试数据的大小
    train_size = int(train_frac * len(trainset))
    test_size = int(test_frac * len(testset))

    # 使用random_split切分数据
    _, trainset = random_split(trainset, [len(trainset) - train_size, train_size])
    _, testset = random_split(testset, [len(testset) - test_size, test_size])

    # 创建DataLoader
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)

    # print(len(train_loader))

    num_users = history_u_lists.__len__()
    num_items = history_v_lists.__len__()
    num_ratings = ratings_list.__len__()

    u2e = nn.Embedding(num_users, embed_dim).to(device)
    v2e = nn.Embedding(num_items, embed_dim).to(device)
    r2e = nn.Embedding(num_ratings, embed_dim).to(device)

    # user feature
    # features: item * rating
    agg_u_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=True)
    enc_u_history = UV_Encoder(u2e, embed_dim, history_u_lists, history_ur_lists, agg_u_history, cuda=device, uv=True)
    # neighobrs
    agg_u_social = Social_Aggregator(lambda nodes: enc_u_history(nodes).t(), u2e, embed_dim, cuda=device)
    enc_u = Social_Encoder(lambda nodes: enc_u_history(nodes).t(), embed_dim, social_adj_lists, agg_u_social,
                           base_model=enc_u_history, cuda=device)

    # item feature: user * rating
    agg_v_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=False)
    enc_v_history = UV_Encoder(v2e, embed_dim, history_v_lists, history_vr_lists, agg_v_history, cuda=device, uv=False)

    # model
    # TODO: graphRec
    graphrec = GraphRec(enc_u, enc_v_history, r2e).to(device)
    optimizer = torch.optim.RMSprop(graphrec.parameters(), lr=args.lr, alpha=0.9)

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0
    rmseArray = []
    maeArray = []
    print("train loader length" + str(len(train_loader)))
    print("test loader length" + str(len(test_loader)))
    for epoch in range(1, args.epochs + 1):

        train(graphrec, device, train_loader, optimizer, epoch, best_rmse, best_mae)
        #torch.save(graphrec.state_dict(), 'ModelSets/testModel' + str(epoch) + '.pth')
        #print("save the model successfully")
        expected_rmse, mae = test(graphrec, device, test_loader)
        # please add the validation set to tune the hyper-parameters based on your datasets.

        # early stopping (no validation set in toy dataset)
        if best_rmse > expected_rmse:
            best_rmse = expected_rmse
            best_mae = mae
            endure_count = 0
        else:
            endure_count += 1
        print("rmse: %.4f, mae:%.4f " % (expected_rmse, mae))
        rmseArray.append(expected_rmse)
        maeArray.append(mae)
        # save the model
        if endure_count > 5:
            break
# TODO: plot the loss function
    plt.figure(figsize=(10, 5))
    plt.plot(rmseArray)
    # "RMSE" + str(count) + "lr:" + str(lr)
    plt.title("RMSE over time")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.show()

    # 绘制 MAE 图
    plt.figure(figsize=(10, 5))
    plt.plot(maeArray)
    plt.title("MAE over time")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.show()
if __name__ == "__main__":
    main()
# TODO : Data visualization dataset
# TODO : Result visualization attention map
# TODO : Use more datasets http://www.cse.msu.edu/~tangjili/trust.html
# TODO : Idea: 1. concat 2 add 2. Normal NN 2 ResNet 3. User Aggregation and social
# TODO : Idea : 4. cross Attention (a friend of mine buy something)
# TODO : Try different datasets C and Epinions 0.6/0.8
# TODO : Compare with other models mentional in the paper
# TODO : Creat new flow graph with slides or something
# TODO : Intro : paper citation
# TODO : Mark everyday's work in github
# TODO : Learn how to code pyTorch copilot/chatGPT
# TODO : Structure : 1. Abstract 2. Intro from graph to GNN to RecSys 3. Prelimelary research
# TODO : 4. Content : proposed model 5. Experiment 6. Conclusion