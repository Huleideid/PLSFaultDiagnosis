# 导入相关库
import numpy as np
from scipy.io import loadmat
import scipy.stats
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power


def autos(X):
    m = X.shape[0]
    n = X.shape[1]
    X_m = np.zeros((m, n))
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    for i in range(n):
        a = np.ones(m) * mu[i]
        X_m[:, i] = (X[:, i]-a) / sigma[i]
    return X_m, mu, sigma

def autos_test(data,m_train,v_train):
    m = data.shape[0]
    n = data.shape[1]
    data_new = np.zeros((m, n))
    for i in range(n):
        a = np.ones(m) * m_train[i]
        data_new[:, i] = (data[:, i] - a) / v_train[i]
    return data_new

def pls_nipals(X, Y, A, max_iter=2000, epsilon=1e-10):
        olv = A
        rankx = np.linalg.matrix_rank(X)
        if olv >= rankx:
            A = rankx

        ssqx = np.sum(X**2)
        ssqy = np.sum(Y**2)
        ssq = np.zeros((A,2))
        ssqdiff = np.zeros((A,2))
        t_old = 0
        iters = 0
        u = Y[:, 0].reshape(Y.shape[0], 1)
        while iters < max_iter:
            W = X.T @ u / (np.linalg.norm(X.T @ u))
            W = W/np.linalg.norm(W)
            T = X @ W
            Q = Y.T @ T / (T.T @ T)
            Q=Q/np.linalg.norm(Q)
            u = Y @ Q
            t_diff = T - t_old
            t_old = T
            if np.linalg.norm(t_diff) < epsilon:
                P = X.T @ T / (T.T @ T)
                X = X - T @ (P.T)
                B = u.T@T/(T.T@T)
                Y = Y-B[0,0]*T@Q.T
                break
            else:
                iters += 1
        R = W

        ssq[0,0] = np.sum(X**2)*100/ssqx;
        ssq[0,1] = np.sum(Y**2)*100/ssqy;

        for i in range(1,A):
            t_old = 0
            iters = 0
            u = Y[:,0].reshape(Y.shape[0],1)
            while iters < max_iter:
                w = X.T @ u / (np.linalg.norm(X.T @ u))
                w = w/np.linalg.norm(w)
                t = X @ w
                q = Y.T @ t / (t.T @ t)
                q = q/np.linalg.norm(q)
                u = Y @ q
                t_diff = t - t_old
                t_old = t
                if np.linalg.norm(t_diff) < epsilon:
                    p = X.T @ t / (t.T @ t)
                    X = X - t @ (p.T)
                    # p=p/np.linalg.norm(p)
                    b = u.T@t/(t.T@t)
                    Y = Y-b[0,0]*t@q.T
                    # t_old = t
                    # r = np.identity(X.shape[1])
                    # for i in range(W.shape[1]):
                    #     # print(W[:,i:i+1]@(P[:,i:i+1].T), X.shape[1])
                    #     r = r @ (np.identity(X.shape[1])- W[:,i:i+1]@(P[:,i:i+1].T))
                    # r = r @ w
                    T = np.hstack((T,t))
                    W = np.hstack((W,w))
                    Q = np.hstack((Q,q))
                    P = np.hstack((P,p))
                    B = np.hstack((B,b))
                    # R = np.hstack((R,r))
                    break
                else:
                    iters += 1
            ssq[i,0] = np.sum(X**2)*100/ssqx;
            ssq[i,1] = np.sum(Y**2)*100/ssqy;

        ssqdiff[0,0] = 100 - ssq[0,0];
        ssqdiff[0,1] = 100 - ssq[0,1];
        ssqdiff[1:,:] = ssq[0:-1,:]-ssq[1:,:]
        R = W @ np.linalg.inv((P.T @ W))
        return T, W, Q, P, R, B, ssqdiff, ssq

class PLSFaultDiagnosis:
    """
    PLS for FaultDiagnosis
    """
    def __init__(self, A=None, kfold=5, max_iter=5000, epsilon = 1e-10, random_state=2022):
        self.A = None
        self.max_iter = max_iter
        self.epsilon = 1e-10
        self.kfold = kfold
        self.random_state = random_state
        self.model = None
        self.VIPs = None
        self.gama = None



    def fit(self, X, y):
        if not self.A:
            self.A = self.cv(X, y, self.kfold) + 1
        print(self.A)
        T,W,Q,P,R,B,ssqdiff,ssq = pls_nipals(X, y,self.A,self.max_iter,self.epsilon)
        self.model = {'T': T, 'W': W, 'Q': Q, 'P': P, 'R': P, 'B':B}
        self.vip(X)  # 计算得到VIPs属性
        lambda_ = (fractional_matrix_power((T.T@T/T.shape[0]), -1))
        lambda_ = np.diag(np.diag(lambda_))
        self.gama = np.real(fractional_matrix_power(R@lambda_@R.T, 1/2))

    def predict(self, X_test):
        T_test_pred = X_test @ self.model['R'];
        Y_test_pred = T_test_pred @ np.diag(self.model['B'][0])@ self.model['Q'].T;
        return Y_test_pred

    def vip(self, X):
        _, p = X.shape
        T = self.model['T']
        Q = self.model['Q']
        W = self.model['W']
        n, h = T.shape
        s=np.diag(T.T@T@Q.T@(Q))
        VIPs = np.zeros(p)
        for i in range(p):
            weight = np.zeros(h)
            for j in range(h):
                weight[j]=(W[i, j]**2)
            q = s.T@np.array(weight)
            VIPs[i] = np.sqrt(p*q/np.sum(s))
        self.VIPs = VIPs

    def contribution(self, X_test):
        cont = self.gama @ X_test.T
        return cont

    def cv(self, X, Y, kfold):
        from sklearn.model_selection import KFold
        from sklearn.metrics import mean_squared_error
        from numpy import mean
        kfold = KFold(n_splits = kfold, shuffle = True, random_state = 2022)
        scores = []
        for pc_num in range(X.shape[1]//2+1):
            cv_scores = []
            for fold, (trn_ind, val_ind) in enumerate(kfold.split(X, Y)):
                X_train, X_test = X[trn_ind], X[val_ind]
                Y_train, Y_test = Y[trn_ind], Y[val_ind]
                T, W, Q, P, R, B, ssqdiff, ssq = pls_nipals(X_train, Y_train, pc_num+1, self.max_iter, self.epsilon)
                T_test_pred = X_test @ R;
                Y_test_pred = T_test_pred @ np.diag(B[0])@ Q.T;
                cv_score=mean_squared_error(Y_test, Y_test_pred)
                cv_scores.append(cv_score)
            scores.append(mean(cv_scores))
        best_A = np.argmin(scores)
        return best_A
    

## 读取数据
path_train = r'.\data\d00.mat'
path_test = r'.\data\d01te.mat'
data1 = loadmat(path_train)['d00']
X1 = data1[:,:22]
X2 = data1[:,-11:]
X_Train= np.hstack((X1,X2))
Y_Train = data1[:,34:35]

data2 = loadmat(path_test)['d01te']
X11 = data2[:,:22]
X22 = data2[:,-11:]
X_test = np.hstack((X11,X22))
# Y_test  = data2[:,34:36]
Y_test  = data2[:,34:35]

# 数据标准化
##训练数据标准化
X_train,X_mean,X_s = autos(X_Train)
Y_train,Y_mean,Y_s = autos(Y_Train)
##测试数据标准化
X_test = autos_test(X_test,X_mean,X_s)
Y_test = autos_test(Y_test,Y_mean,Y_s)

## 新建类
pls = PLSFaultDiagnosis()

## 拟合
pls.fit(X_train, Y_train)

# 预测
Y_test_predict = pls.predict(X_test)

# 预测杰结果可视化
plt.figure()
plt.plot(Y_test_predict,color="r")
plt.plot(Y_test,color="b")
plt.show()

# 贡献图
# cont = pls.contribution(X_test)
cont = pls.contribution(X_test[200,:]).ravel()
# np.argsort(-cont)[:3]

# 贡献图可视化
import matplotlib.pyplot as plt
plt.figure()
plt.bar(np.arange(1, len(cont)+1, 1),np.abs(cont))
plt.show()