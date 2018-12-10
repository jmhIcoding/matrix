#coding:utf-8
__author__ = 'jmh081701'
import  numpy as np
import  copy
import  sys

np.set_printoptions(precision=3,suppress=True)
def print_matrix(mat):
    shape=np.shape(mat)
    for i in range(shape[0]):
        for j in range(shape[1]):
            print(round(mat[i,j],3),end="\t")
        print(end="\n")

def LU_decompose(mat):
    mat=copy.deepcopy(mat)
    shape = np.shape(mat)
    if shape[0]!=shape[1]:
        print("The matrix is not a square matrix,which cannot be LU decomposed.")
        return None,None,None
    P=np.zeros(shape)
    U=np.eye(shape[0])
    L=np.eye(shape[0])
    per=[i for i in  range(shape[0])] #记录
    for i in range(shape[0]):
            #主元为0,需要进行交换
            j_max=i
            for j in range(i+1,shape[0]):
                if mat[j,i] != 0:
                    #向下找到非0绝对值最大的主元,交换上去
                    if abs(mat[j,i]) > abs(mat[j_max,i]):
                        j_max = j

            tmp = copy.deepcopy(mat[i,:])
            mat[i,:] = copy.deepcopy(mat[j_max,:])
            mat[j_max,:] = tmp
            #交换per记录
            tmp = per[i]
            per[i]=per[j_max]
            per[j_max]=tmp
            if mat[i,i]==0:
                print("This matrix is single,which cannot be decomposed")#无法被LU分解
                return  None,None,None
            for j  in range(i+1,shape[0]):
                if mat[j,i]!=0:
                    factor = mat[j,i]/mat[i,i]
                    mat[j,i:]=mat[j,i:]-factor * mat[i,i:]
                    mat[j,i]=factor
                else:
                    continue
    #构建解
    #P矩阵
    for i in range(shape[0]):
        P[i,per[i]]=1
    #L矩阵
    for i in range(shape[0]):
        U[i,i:]=mat[i,i:]
        if i>0:
            L[i,0:i]=mat[i,0:i]
    print("LU decompose result.")
    print('P:')
    print_matrix(P)
    print("L:")
    print_matrix(L)
    print("U:")
    print_matrix(U)
    return np.asmatrix(P),np.asmatrix(L),np.matrix(U)
def QR_decompose(mat):
    mat =copy.deepcopy(mat)
    mat = np.asmatrix(mat,dtype=np.float)
    shape = np.shape(mat)
    R= np.eye(shape[1])
    for column in range(shape[1]):
            for i in  range(column):
                mat[:,column] = np.asmatrix(mat[:,column])
                project =np.float( (mat[:,i]).T * mat[:,column])
                R[i,column] =project
                for row in range(shape[0]):
                    mat[row,column] -=project * mat[row,i]

            mat[:,column]=np.asmatrix(mat[:,column])#先转换为矩阵
            magnitude=np.float(np.sqrt((mat[:,column]).T * mat[:,column]))
            if magnitude==0:
                print("This Matrix cannot be QR decomposed.")
                return None,None,None
            for row in range(shape[0]):
                mat[row,column] /=magnitude
            R[column,column]=magnitude
    print("QR decompose result.")
    print("Q:")
    print_matrix(mat)
    print('R:')
    print_matrix(R)
    return np.asmatrix(mat),np.asmatrix(R)

def HouseholderReduction(mat):
    mat = np.asmatrix(mat,dtype=np.float)
    shape = mat.shape
    R=np.eye(shape[0])
    T=np.zeros(shape=shape)
    for i in range(shape[0]):
        submat = np.asmatrix(mat[i:,i:])
        #把第一列取出来
        x= np.asmatrix(submat[:,0])
        if(x.shape[0]==1):
            continue
        u=copy.deepcopy(x)
        u[0,0]-=np.sqrt(np.float(x.T * x)) #
        print(u.T *u)
        _Ri=np.eye(u.shape[0])-2*u*u.T/(np.float(u.T * u))
        Ri=np.eye(shape[0])
        Ri[i:,i:]=_Ri
        Ri= np.asmatrix(Ri)
        R =Ri *R
        mat=np.asmatrix(mat)
        mat=Ri * mat
    print("House Holder Reduce result.")
    print('P:')
    print_matrix(R.T)
    print('T:')
    print_matrix(mat)
    return np.asmatrix(R,dtype=np.float),np.asmatrix(mat,dtype=np.float)
def GivensReduction(mat):
    mat = np.asmatrix(mat,dtype=np.float)
    shape = mat.shape
    P=np.eye(shape[0])
    T=np.zeros(shape=shape)
    for i in range(shape[0]):
        column= np.asmatrix(mat[:,i])
        #把第i列取出来
        if(column.shape[0]==1):
            #第i列只有一个元素了,没必要再约减
            continue
        u=copy.deepcopy(column)
        for row in range(i+1,shape[0]):
            mag=np.sqrt((u[row,0]**2+u[i,0]**2))
            if(mag)==0:
                #自带0行,不用约减
                continue
            c=u[i,0] / mag
            s=u[row,0]/mag
            Pi=np.eye(shape[0])
            Pi[i,i]=c
            Pi[row,row]=c
            Pi[i,row]=s
            Pi[row,i]=-s
            Pi = np.asmatrix(Pi)
            P=Pi*P
            u=Pi*u
            mat=np.asmatrix(mat)
            mat=Pi*mat
    print("Givens Reduce result.")
    print('P')
    print_matrix(P.T)
    print('T')
    print_matrix(mat)
    return np.asmatrix(P,dtype=np.float),np.asmatrix(mat,dtype=np.float)
if __name__ == '__main__':
    #A=np.asmatrix([[0,-20,-14],[3,27,-4],[4 ,11,-2]])
    #HouseholderReduction(A)
    while True:
        print(">============================================<")
        cmd = input("please select one from this four method :\n LU(1)\t QR(2)\t HouseHolder Reduction(3)\t Givens Reduction(4) End(5):")
        func=None
        if "1" in cmd:
            print("Your select LU decompose method.")
            func=LU_decompose
        elif "2" in cmd:
            print("Your select QR decompose method.")
            func=QR_decompose
        elif "3" in cmd:
            print("Your select HouseHolder Reduction method.")
            func=HouseholderReduction
        elif "4" in cmd:
            print("Your select Givens Reduction method.")
            func=GivensReduction
        elif "5" in cmd:
            print("Bye···"*3)
            break
        else:
            print("Unknow command.")
            continue
        row_size=input("Please input the matrix size(for example 4 4)\n\trow size:")
        col_size=input("\tcol size:")
        mat=[]
        row_size=int(row_size)
        col_size=int(col_size)
        for i in range(row_size):
            raw_row=input("The %d row, each entity seperate by space:"%i)
            row = raw_row.split(" ")
            row = [float(each) for each in row]
            mat.append(row)
        mat = np.asmatrix(mat,dtype=np.float)
        func(mat)