# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter

if __name__ == '__main__':
    print('\n'*10)
    
    iris = datasets.load_iris() 
    print(type(iris), dir(iris))

    x = iris.get('data')
    y = iris.get('target')

    # show attributes histogram
    c = np.unique(y)
    ind = []
    ind.append(y==c[0])
    ind.append(y==c[1])
    ind.append(y==c[2])
    bin_num = 40
    fig, axes = plt.subplots(len(c),4)
    for i, ax in enumerate(axes.flat):
        ind_ = ind[i//4]
        j = i%4
        ax.hist(x[ind_,j], bins=bin_num)

    axes[0,0].set_ylabel("y = 0")
    axes[1,0].set_ylabel("y = 1")
    axes[2,0].set_ylabel("y = 2")
    axes[0,0].set_title("attribute 0")
    axes[0,1].set_title("attribute 1")
    axes[0,2].set_title("attribute 2")
    axes[0,3].set_title("attribute 3")
    plt.show()

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

    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_test_pre = gnb.predict(x_test)

    # 计算分类准确率
    acc = sum(y_test_pre==y_test)/num_test
    print('the accuracy is', acc) # 显示预测准确率

    
    
