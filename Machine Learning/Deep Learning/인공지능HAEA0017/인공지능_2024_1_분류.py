import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression

# [data 준비 과정]

# data loading

# frature selection

# data cleansing
# 	missing value
# 	duplicate
# 	anomaly

# 변수 설정 

# 변수 벡터 설정

#  train-test split

nbastat = pd.read_csv("data/csv/nba_stats_2020_to_2023.csv")

m = len(nbastat)

features = ['Pos', 'FG%', '3P%', 'TRB', 'AST', 'STL', 'BLK']
nbastat2 = nbastat[features]

rows_with_na = nbastat2[nbastat2.isna().any(axis=1)]

nbastat2 = nbastat2.dropna(axis=0)

nbastat3 = nbastat2[(nbastat2['Pos'] == 'C') | (nbastat2['Pos'] == 'PG') | (nbastat2['Pos'] == 'SG') | (nbastat2['Pos'] == 'SF') | (nbastat2['Pos'] == 'PF')]

y = nbastat3[['Pos']]
x1 = nbastat3[['FG%']]
x2 = nbastat3[['3P%']]
x3 = nbastat3[['TRB']]
x4 = nbastat3[['AST']]
x5 = nbastat3[['STL']]
x6 = nbastat3[['BLK']]

#이상치 제거 확인
unique = y.drop_duplicates()
print(unique)

yb = nbastat3['Pos'].apply(lambda x : 1 if x == 'C' else 0)

yt = nbastat3['Pos'].map({'C':2, 'PF':1, 'SF':1, 'PG':0, 'SG':0})


m = len(x1)
x0 = np.ones((m,1))
xb = np.c_[x0, x1, x2, x3, x4, x5, x6]
yb = np.array(yb).reshape(m,1)

def permutation_split(X, Y, ratio=0.7, random_state=1004):
    num_a = int(X.shape[0] * ratio)
    num_b = X.shape[0] - num_a

    np.random.seed(random_state)
    shuffle = np.random.permutation(X.shape[0])

    X = X[shuffle, :]
    Y = Y[shuffle, :]

    Xa = X[:num_a]
    Ya = Y[:num_a]
    Xb = X[num_a:]
    Yb = Y[num_a:]

    return Xa, Xb, Ya, Yb

Xb_train, Xb_test, Y_train, Y_test = permutation_split(xb, yb, 0.6)

class LogisticRegressionNumpy:
    def __init__(self, lr = 0.0001, epoch = 1000):
        self.lr = lr
        self.epoch = epoch
        self.theta = None

    def loss_CE(self, y_hat, y):
        epsilon = 1e-15
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    
    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z))

    def train(self, X, y):
        m, n = X.shape
        self.theta = np.zeros((n,1))
        loss_arr = []

        for _ in range(self.epoch):
            Z = X.dot(self.theta)
            y_hat = self.sigmoid(Z)

            loss_arr.append(self.loss_CE(y_hat, y))

            grad = (1/m) * X.T.dot(y_hat - y)

            self.theta = self.theta - self.lr * grad
        
        return loss_arr

    def predict(self, X):
        Z = X.dot(self.theta)
        y_hat = self.sigmoid(Z)

        y_hat_cls = [1 if i > 0.5 else 0 for i in y_hat]
        return y_hat_cls
    

# model = LogisticRegressionNumpy(0.01, 10000)
# loss_arr = model.train(Xb_train, Y_train)

# plt.plot(loss_arr, '.b')
# plt.ylabel("loss")
# plt.show()

# prediction = model.predict(Xb_train)
# cnt = 0
# for i in range(len(prediction)):
#     cnt += prediction[i] == Y_train[i]

# print(100 * (cnt / len(prediction)))

# prediction = model.predict(Xb_test)
# cnt = 0
# for i in range(len(prediction)):
#     cnt += prediction[i] == Y_test[i]

# print(100 * (cnt / len(prediction)))






device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LogisticRegressionPytorch(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegressionPytorch, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, Xb):
        y_hat = self.sigmoid(self.linear(Xb))
        return y_hat

    def predict(self, Xb):
        y_hat = self.sigmoid(self.linear(Xb.to(device))).cpu()
        return [1 if i > 0.5 else 0 for i in y_hat]

def train(model, X, y, lr, epoch):
    critetrian = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr = lr)

    X = torch.tensor(X, dtype = torch.float32).to(device)
    y = torch.tensor(y, dtype = torch.float32).view(-1,1).to(device)

    loss_arr = []
    for _ in range(epoch):
        optimizer.zero_grad()
        output = model(X)
        loss = critetrian(output, y)

        loss.backward()
        loss_arr.append(loss.cpu().detach().numpy())

        optimizer.step()

    return loss_arr

model = LogisticRegressionPytorch(input_size = 7).to(device)
loss_arr = train(model, Xb_train, Y_train, lr = 0.01, epoch = 100000)

plt.plot(loss_arr, '.b')
plt.ylabel("loss")
plt.show()

prediction = model.predict(torch.tensor(Xb_train, dtype=torch.float32))
cnt = 0
for i in range(len(prediction)):
    cnt += prediction[i] == Y_train[i]

print(100 * (cnt / len(prediction)))

prediction = model.predict(torch.tensor(Xb_test, dtype=torch.float32))
cnt = 0
for i in range(len(prediction)):
    cnt += prediction[i] == Y_test[i]

print(100 * (cnt / len(prediction)))



# model = LogisticRegression(max_iter = 1000)

# model.fit(Xb_train, Y_train.ravel())

# prediction = model.predict(Xb_train)
# cnt = 0
# for i in range(len(prediction)):
#     cnt += prediction[i] == Y_train[i]

# print(100 * (cnt / len(prediction)))

# prediction = model.predict(Xb_test)
# cnt = 0
# for i in range(len(prediction)):
#     cnt += prediction[i] == Y_test[i]

# print(100 * (cnt / len(prediction)))