# parameters

import pandas as pd
import math
import random
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rc('font', family='Malgun Gothic')

average = 0

phi = [ ] # 구하고자하는 매개 변수
matrix_A = [ ] # A * X(phi) = B
matrix_B = [ ]
matrix_A_inv = [ ] # A 역행렬
time_series = [ ]  # 시계열 데이터
p_time_series = [ ]  # 예상 시계열 데이터
upper_limit = [ ] # 예측 구간
lower_limit = [ ]
predict_variances = [ ]
residual = [ ] # 잔차
fittedvalues = [] #적합값
pred =[] #예측값

# 가상 데이터 생성( sin(x) )

def v_data () :
    series = [0] * n
    series[0:] = [math.sin(math.radians(30) * i) for i in range (n)]

    return series


# 평균 계산

def E (mat) :
    s = 0
    for i in range(len(mat)) :
        s += mat[i]
    return s / len(mat)

# 자기공분산

def r (k, mat) :
    sum = 0
    if k >= 0 :
        for i in range (k, len(mat), 1) :
            sum += (mat[i] - E(mat)) * (mat[i - k] - E(mat))
        return sum / len(mat)
    else :
        k *= -1
        for i in range (k, len(mat), 1) :
            sum += (mat[i] - E(mat)) * (mat[i - k] - E(mat))
        return sum / len(mat)



# 역행렬 계산(가우스-조르단)
    
def matrix_inv (matrix) :

    #단위 행렬 생성
    
    matrix_id = [ ]
    for i in range(0, len(matrix) , 1) :
        line = [ ]
        for x in range(0, len(matrix), 1) :
            if i == x :
                line.append(1)
            else :
                line.append(0)   
        matrix_id.append(line)

    #계산

    for w in range(1, len(matrix) + 1, 1) :
        index =[ ]
        for i in range(w-1, len(matrix), 1) :
            index.append( matrix[i][w-1] )
            if index[i-w+1] == 0 :
                continue
            else :
                for x in range(len(matrix)) :
                    matrix[i][x] /= index[i-w+1]
                    matrix_id[i][x] /= index[i-w+1]

        if index[0] == 0 :
            for y in range(w-1, len(matrix) - 1, 1) :
                    if index[y-(w-1)+1] == 0 :
                        continue
                    else :
                        for z in range(len(matrix)) :
                            matrix[w-1][z] += matrix[y+1][z]
                            matrix_id[w-1][z] += matrix_id[y+1][z]
                        break
            
        if w < len(matrix) :
            for i in range(w,len(matrix),1) :
                if index[i-w+1] == 0 :
                    continue
                else :
                    for x in range(len(matrix)) :
                        matrix[i][x] -= matrix[w-1][x]
                        matrix_id[i][x] -= matrix_id[w-1][x]
        

    for w in range(len(matrix) - 1, 0 , -1) :
        for i in range(w-1, -1 , -1) :
            index = matrix[i][w]
            matrix[i][w] -= matrix[w][w]*index
            for x in range(len(matrix)) :
                matrix_id[i][x] -= matrix_id[w][x]*index
    
    return matrix_id

# 자기 상관계수

def corr (mat, k) :
    return (r(k, mat) / r(0, mat))

# 부분 자기 상관계수

def pacf(mat, k) :
    if k == 0:
        return 1

    else :
        A = [ ]
        for i in range(k) :
            matrix = [0] * k
            matrix[:] = [corr(mat, j-i) for j in range(k)]
            A.append(matrix)

        B =[0] * k
        B[:] = [corr(mat, j) for j in range(1,k + 1)]

        A_inv = matrix_inv(A)

        pacf_val = [ ]
        for i in range(k) :
            sum = 0
            for x in range(k) :
                sum += A_inv[i][x] * B[x]
            pacf_val.append(sum)
        return pacf_val[-1]

# white noise 분산

def w_variance( ) :
    sum = 0
    for i in range(p) :
            sum += phi[i] * r( i+1,time_series)
    return (r(0, time_series) - sum) 



# 예측값

def predict(mat) :

    k = int(input(' 몇 개의 데이터를 예측하겠습니까?: '))
    for i in range(p):
        p_time_series.append(mat[i])
        fittedvalues.append(mat[i])

    alpha = 0.05
    
    # intercept 계산
    sum = 0
    for i in range(p) :
        sum += phi[i]
    c = (1 - sum) * average 


    # 적합값
    for w in range(p, len(mat), 1) :   
        sum1 = c 
        for i in range(p) :
            sum1 += phi[i] * (time_series[(w)-(i+1)])
        
        p_time_series.append(sum1)
        fittedvalues.append(sum1)

    #잔차 계산
    for i in range(len(mat)) :
        residual.append(p_time_series[i] - time_series[i])

        
    # 예측값
    
    for w in range(k) :   
        sum2 = c 
        for i in range(1,p+1,1) :
            if len(mat)+w-i >= len(mat) :
                sum2 += phi[i-1] * (p_time_series[len(mat)+w-i])
            else :
                sum2 += phi[i-1] * (time_series[len(mat)+w-i] )
        p_time_series.append(sum2)
        pred.append(sum2)
        upper_limit.append(sum2  + norm.ppf(1 - alpha / 2) * math.sqrt(w_variance( )))
        lower_limit.append(sum2  - norm.ppf(1 - alpha / 2) * math.sqrt(w_variance( )))


                       
    # 그래프 구현

    index_k_days = pd.date_range('2022-06-04', freq='D', periods = k, tz = None)

    fig , ax = plt.subplots(figsize=(7,5))
    ax.vlines(index_k_days[0],200000,400000,linestyle='--', color='r', label='Start of Forecast')
    plt.title('PREDICTION')
    plt.plot(df['Datetime'],mat,color='blue',linewidth=2, label='data')
    plt.plot(df['Datetime'],fittedvalues,color='red',linewidth=1, label = 'fittedvalues')
    plt.plot(index_k_days,pred,color='green',linewidth=1, label = 'prediction')
    ax.fill_between(index_k_days,upper_limit,lower_limit, color = 'k', alpha = 0.1, label = 'conf_int(95%)')
    plt.legend()
    plt.show()

############################################################################################################# <<< 실행부

# 시계열


df = pd.read_excel('naver.xlsx')

df['Datetime'] = pd.to_datetime(df['날짜'])
data = df['종가'].values

for i in range(len(data)) :
    time_series.append(data[i])


"""
n = int(input(' 몇 개의 데이터를 만들겠습니까?: '))
time_series = v_data()
"""

# 평균 계산

average = E(time_series)


# 부분 자기 함수 확인 및 p 설정

pacf_len = 10   # 확인하는 pacf 개수
pacfs = [pacf(time_series,k) for k in range(pacf_len)]
pacf_x = range(pacf_len)
acfs = [corr(time_series,k) for k in range(pacf_len)] ##

##ACF
plt.subplot(2,1,1)
markers, stemlines, baseline = plt.stem(pacf_x,acfs)
plt.title('ACF')
markers.set_color('red')
stemlines.set_linestyle('--')
stemlines.set_color('purple')

##PACF
plt.subplot(2,1,2)
markers, stemlines, baseline = plt.stem(pacf_x,pacfs)
plt.title('PACF')
markers.set_color('red')
stemlines.set_linestyle('--')
stemlines.set_color('purple')
#baseline.set_visible(False)

plt.show(block=False)
plt.pause(5)
plt.close()

p = int(input(' 몇 개 전의 데이터까지 고려하겠습니까?: '))



# 행렬 A 계산

for l in range(p) :
    list = [ ]
    for k in range(p) :
        list.append(r(l-k,time_series))
    matrix_A.append(list)


# 행렬 A의 역행렬 계산

matrix_A_inv = matrix_inv(matrix_A)


# 행렬 B 계산

for i in range(p) :
    matrix_B.append(r(i+1,time_series))


# 매개 변수 계산

for i in range(p) :
    sum = 0
    for x in range(p) :
        sum += matrix_A_inv[i][x] * matrix_B[x]
    phi.append(sum)


# 추측

predict(time_series)


# 잔차 ACF 확인

res_acfs = [corr(residual,k) for k in range(pacf_len)] ##

##ACF
markers, stemlines, baseline = plt.stem(pacf_x,res_acfs)
plt.title('RES_ACF')
markers.set_color('red')
stemlines.set_linestyle('--')
stemlines.set_color('purple')


plt.plot(pacf_x, [3 * r(0,res_acfs)] * pacf_len ,color = 'dodgerblue') 
plt.plot(pacf_x, [3 * r(0,res_acfs) * -1] * pacf_len ,color = 'dodgerblue')

plt.show()
#################################################################################################################










    
