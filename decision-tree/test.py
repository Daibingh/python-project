# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris

if __name__ == '__main__':
    print('\n\n\n\n\n\n\n\n\n\n')

    # show data info
    data = load_iris()
    print('keys: \n', data.keys()) # ['data', 'target', 'target_names', 'DESCR', 'feature_names']
    feature_names = data.get('feature_names')
    print('feature names: \n', data.get('feature_names'))
    print('target names: \n', data.get('target_names'))
    x = data.get('data')
    y = data.get('target')
    # print(x.shape, y.shape)
    # print(x)
    # print(data.get('DESCR'))

    # visualize the data

    f = []
    f.append(y==0)
    f.append(y==1)
    f.append(y==2)
    color = ['red','blue','green']
    fig, axes = plt.subplots(4,4)
    for i, ax in enumerate(axes.flat):
        row  = i // 4
        col = i % 4
        if row == col:
            ax.text(.1,.5, feature_names[row])
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        for  k in range(3):
            ax.scatter(x[f[k],row], x[f[k],col], c=color[k], s=3)    
    fig.subplots_adjust(hspace=0.3, wspace=0.3) # 设置间距
    # plt.show()

    # 划分训练集和测试集
    num = x.shape[0]
    ratio = 7/3 # 训练集数目：测试集数目
    num_test = int(num/(1+ratio))
    num_train = num -  num_test
    index = np.arange(num)
    np.random.shuffle(index)
    x_test = x[index[:num_test],:]
    y_test = y[index[:num_test]]
    x_train = x[index[num_test:],:]
    y_train = y[index[num_test:]]
    
    # 构建决策树
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)

    # 预测
    y_test_pre = clf.predict(x_test)
    print('the predict values are', y_test_pre)

    # 计算分类准确率
    acc = sum(y_test_pre==y_test)/num_test
    print('the accuracy is', acc)