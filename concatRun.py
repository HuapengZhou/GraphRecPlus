import collections

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, random_split
import pickle
import numpy as np
import time
import random
from collections import defaultdict
from VC_Encoder_Aggregator import VC_Aggregator, VC_Encoder
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

        x_uv = torch.cat((x_u, x_v), 1)  # TODO: concat changes +
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


def the_test(model, device, test_loader):
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
    parser.add_argument('--embed_dim', type=int, default=8, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
        print("I am using cuda")  # my own add
    device = torch.device("cuda" if use_cuda else "cpu")

    embed_dim = args.embed_dim
    # history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, social_adj_lists, ratings_list = pickle.load(
    #     data_file)
    # Load the data
    with open('./data/Ciao80_with_cat.pickle', 'rb') as data_file:
        data = pickle.load(data_file)

    history_u_lists = data[0]
    history_ur_lists = data[1]
    history_v_lists = data[2]
    history_vr_lists = data[3]
    history_vc_lists = data[4]  # additional category information
    train_u = data[5]
    train_v = data[6]
    train_r = data[7]
    test_u = data[8]
    test_v = data[9]
    test_r = data[10]
    social_adj_lists = data[11]
    ratings_list = data[12]

    num_categories = 10
    # Create datasets
    # print(train_u)
    # print(train_v)
    # print(train_r)
    # train_u = train_u.astype(np.int64)
    # train_v = train_v.astype(np.int64)
    # train_r = train_r.astype(np.float32)
    # test_u = test_u.astype(np.int64)
    # test_v = test_v.astype(np.int64)
    # test_r = test_r.astype(np.float32)

    # Create datasets
    trainset = TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v), torch.FloatTensor(train_r))
    testset = TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v), torch.FloatTensor(test_r))
    print(len(trainset))
    # Data split
    train_size = int(0.01 * len(trainset))
    test_size = int(0.1 * len(testset))

    _, trainset = random_split(trainset, [len(trainset) - train_size, train_size])
    _, testset = random_split(testset, [len(testset) - test_size, test_size])

    # Create DataLoader
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)

    num_users = len(history_u_lists)
    num_items = len(history_v_lists)
    num_ratings = len(ratings_list)

    # Embedding layers
    u2e = nn.Embedding(num_users, embed_dim).to(device)
    v2e = nn.Embedding(num_items, embed_dim).to(device)
    r2e = nn.Embedding(num_ratings, embed_dim).to(device)
    c2e = nn.Embedding(num_categories, embed_dim).to(device)  # additional category embedding

    # User feature: item * rating
    agg_u_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device)
    enc_u_history = UV_Encoder(u2e, embed_dim, history_u_lists, history_ur_lists, agg_u_history, cuda=device)

    # Neighbors
    agg_u_social = Social_Aggregator(lambda nodes: enc_u_history(nodes).t(), u2e, embed_dim, cuda=device)
    enc_u = Social_Encoder(lambda nodes: enc_u_history(nodes).t(), embed_dim, social_adj_lists, agg_u_social,
                          base_model=enc_u_history, cuda=device)

    # Item feature: user * rating
    agg_v_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=False)
    enc_v_history = UV_Encoder(v2e, embed_dim, history_v_lists, history_vr_lists, agg_v_history, cuda=device, uv=False)

    # Category feature: item * rating
    agg_v_category = VC_Aggregator(c2e, r2e, v2e, embed_dim, cuda=device)
    enc_v_category = VC_Encoder(v2e, embed_dim, history_vc_lists, history_vr_lists, agg_v_category, cuda=device)

    # Model
    graphrec = GraphRec(enc_u, enc_v_history, r2e).to(device)  # assuming GraphRec can handle the additional VC structure
    optimizer = torch.optim.RMSprop(graphrec.parameters(), lr=args.lr, alpha=0.9)

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0
    rmseArray = []
    maeArray = []
    print("train loader length" + str(len(train_loader)))
    print("test loader length" + str(len(test_loader)))
    # print(graphrec)
    for epoch in range(1, args.epochs + 1):

        train(graphrec, device, train_loader, optimizer, epoch, best_rmse, best_mae)
        # torch.save(graphrec.state_dict(), 'ModelSets/testModel' + str(epoch) + '.pth')
        # print("save the model successfully")
        expected_rmse, mae = the_test(graphrec, device, test_loader)
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