#1. import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
#from google.colab import drive

#2. mount drive
#drive mount("/smuai2024")

#3. upload file
#from google.colab import files
#files.upload()

#4. read csv file
path = os.getenv("RESOURCE_PATH")
nbastat = pd.read_csv(path + "\\csv\\nbastat2022.csv")

#5. 행 수 count
m = len(nbastat) # m: 데이터(sample)의 수 => 249

#6. feature selection => nbastat 에서 column 선택
#이 column들만 사용하여 모델 수행
X = nbastat[["FGA"]]
Y = nbastat[["FGM"]]

print(X.head())


#7. 결측값 처리
#pandas 에서 결측값을 해소하는 해소하는 함수: fillna
X = X.fillna(0)
Y = Y.fillna(0)

#8. ndarray 로 변환
#pandas의 dataframe -> np의 array로 변환
X = np.array(X).reshape(m,1)
Y = np.array(Y).reshape(m,1)

#9. plotting
# plt.plot(X, Y, '.b')
# plt.xlabel("FGA")
# plt.ylabel("FGM")
# plt.show()

#10. hyper parameter
learning_rate = 0.001
epoch = 2000

#11. initialization
# theta 와 gradieent 초기화 -> 0 으로 초기화
theta = np.zeros((2,1))
grad = np.zeros((2,1))

#12. 변수 설정
x0 = np.ones((m,1))
Xb = np.c_[x0, X]

#13. training
for i in range(epoch):
    # 1) Xb@theta --> Xb.dot(theta)
    # 2) Xn@theta - Y --> Xb.dot(theta) - Y
    # 3) Xb.T @ (Xb*theta - Y) --> Xb.T.dot(Xb.dot(theta) - Y)
    grad = (1./m)*Xb.T.dot(Xb.dot(theta) - Y)
    theta = theta - learning_rate * grad

#14. 결과 가시화
# (X, Y)의 데이터와 Y = theta_0 + X * theta_1
Y_pred = Xb.dot(theta)

plt.plot(X, Y_pred, color="Red")
plt.plot(X, Y, '.b')
plt.show()