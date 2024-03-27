import sys

matrix_id = []
answer = 0
#행렬 생성

while answer != 1 :
    matrix = [ ]
    matrix_b = [ ]
    matrix_n = int(input( '몇 by 몇 정사각 행렬인가요?, 숫자 하나만 입력: '))
    for i in range(1, matrix_n + 1, 1) :
        line = input('%d 행 을 입력 해주세요(각 원소는 띄어쓰기): ' % (i) )
        r = [ ]
        k = [ ]
        for x in range(matrix_n) :
            r.append(int(line.split()[x]))
            k.append(int(line.split()[x]))
        matrix.append(r)
        matrix_b.append(k)
            
    print()
    for y in range(matrix_n) : 
        print(matrix[y])
    print()
    answer = int(input('원하는 행렬이 맞습니까? 맞으면 1, 아니면 0 : '))

# 3x3 determinant

def det_3x3(mat) :
    det = 0

    for i in range(3) :
        k = 1
        for x in range(3):
            k *= mat[x][(i+x) % 3]
        det += k

    for i in range(3) :
        k = 1
        for x in range(3):
            k *= mat[x][(2*(x+1)+i) % 3]
        det -= k
    return det

# 2x2 determinant
        
def det_2x2(mat) :
    det = 0

    for i in range(2) :
        k = 1
        for x in range(2) :
            k *= mat[x][(i+x) % 2]
        if i == 0 :    
            det += k
        else :
            det -= k
    return det
        
# 행렬식 계산

def det(mat):
    sum_t = 0
    sum = 0
    if len(mat) > 3 :
        mat_A = [ ]
        for x in range(len(mat)) :
            mat_a = [ ]
            for y in range(1,len(mat), 1) :
                mat_copy = [ ]
                for z in range(len(mat)) :
                    if z == x :
                        continue
                    else :
                        mat_copy.append(mat[y][z])
                        
                mat_a.append(mat_copy)  #n-1 x n-1
            mat_A.append(mat_a)
       
        for i in range(len(mat)) :
            sum_t += det(mat_A[i]) * mat[0][i] * (-1)**i
        return sum_t
    
    elif len(mat) == 3 :
        sum += det_3x3(mat)
        return sum
    
    else :
        sum += det_2x2(mat)
        return sum

#필터

def int_lize(num) :
    z = 0
    df = 0
    if num >= 0 :
        z = int(num)
        df = num - z
        if 1 - df < 10**(-12) :
            num = z + 1.0
            return num
        if df < 10**(-12) :
            num -= df
            return num
        return num
        
    else :
        z = int(num) - 1
        df = num - z
        if 1 - df < 10**(-12) :
            num += 1 - df
            return num
        if df < 10**(-12) :
            num -= df
            return num
        return num
        
#행렬식 확인

if det(matrix) == 0 :
    sys.exit('\n이 함수는 역함수를 가지지 않습니다. 행렬식: %d' % det(matrix))
    
else :
    print('\n행렬식:', det(matrix))
    answer = int(input('\n필터를 키겠습니까? yes면 1, no면 0 : '))

  
#단위 행렬 생성 
        
for i in range(0, matrix_n , 1) :
    line = [ ]
    for x in range(0, matrix_n, 1) :
        if i == x :
            line.append(1)
        else :
            line.append(0)   
    matrix_id.append(line)


#역행렬 계산
    
def matrix_inv (matrix) :
    for w in range(1, matrix_n + 1, 1) :
        index =[ ]
        for i in range(w-1, matrix_n, 1) :
            index.append( matrix[i][w-1] )
            if index[i-w+1] == 0 :
                continue
            else :
                for x in range(matrix_n) :
                    matrix[i][x] /= index[i-w+1]
                    matrix_id[i][x] /= index[i-w+1]
                
        if index[0] == 0 :
            for y in range(w-1, matrix_n - 1, 1) :
                    if index[y-(w-1)+1] == 0 :
                        continue
                    else :
                        for z in range(matrix_n) :
                            matrix[w-1][z] += matrix[y+1][z]
                            matrix_id[w-1][z] += matrix_id[y+1][z]
                        break
            

                        
            
        if w < matrix_n :
            for i in range(w,matrix_n,1) :
                if index[i-w+1] == 0 :
                    continue
                else :
                    for x in range(matrix_n) :
                        matrix[i][x] -= matrix[w-1][x]
                        matrix_id[i][x] -= matrix_id[w-1][x]
        

    for w in range(matrix_n - 1, 0 , -1) :
        for i in range(w-1, -1 , -1) :
            index = matrix[i][w]
            matrix[i][w] -= matrix[w][w]*index
            for x in range(matrix_n) :
                matrix_id[i][x] -= matrix_id[w][x]*index

    if answer == 1 :                    #필터 부분
        for i in range(matrix_n) :
            for w in range(matrix_n) :
                matrix_id[i][w] = int_lize(matrix_id[i][w])
                
    print('\n\n역행렬은: \n')
    for i in range(matrix_n) : 
            print(matrix_id[i])


#검산(정사각 행렬끼리의 백터 곱)
            
def check () :
    matrix_check = [ ]
    for w in range(matrix_n) :
        line = [ ]
        for i in range(matrix_n) :
            k = 0
            for x in range(matrix_n) :
                k += matrix_b[w][x]*matrix_id[x][i]
            line.append(k)
        matrix_check.append(line)

    if answer == 1 :                    #필터 부분
        for i in range(matrix_n) :
            for w in range(matrix_n) :
                matrix_check[i][w] = int_lize(matrix_check[i][w])

    print('\n\n검산 결과: \n')
    for i in range(matrix_n) : 
            print(matrix_check[i])

     
# 실행
matrix_inv(matrix)
check()

 
        


        
    

    
