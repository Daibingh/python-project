from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter

def mode(a):
    counts = np.bincount(a)  
    return np.argmax(counts)

def calc_acc(y_p, y):
    return sum(y_p==y)/y.shape[0]


if __name__ == '__main__':
    
    iris = datasets.load_iris()

    x = iris.get('data')
    y = iris.get('target')

    # 随机划分训练集和测试集
    num = x.shape[0] # 样本总数
    ratio = 7/3 # 划分比例，训练集数目:测试集数目
    num_test = int(num/(1+ratio)) # 测试集样本数目
    num_train = num -  num_test # 训练集样本数目
    index = np.arange(num) # 产生样本标号
    np.random.shuffle(index) # 洗牌
    x_test = x[index[:num_test],:] # 取出洗牌后前 num_test 作为测试集
    y_test = y[index[:num_test]]
    x_train = x[index[num_test:],:] # 剩余作为训练集
    y_train = y[index[num_test:]]

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(x_train)

    centers = kmeans.cluster_centers_
    for i in range(3):
        index = y_train == i
        p = kmeans.predict(x_train[index,:])
        pp = mode(p) # 求实际类别为 i 所对应的类别标号 pp
        kmeans.cluster_centers_[i] = centers[pp]

    y_test_pre = kmeans.predict(x_test)
    print("y_test_pre:")
    print(y_test_pre)
    print("y_test:")
    print(y_test)

    # 计算分类准确率
    acc = calc_acc(y_test_pre, y_test)
    print('the accuracy is', acc) # 显示预测准确率

    
    
