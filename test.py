# -*- coding: utf-8 -*
import pandas as pd
import matplotlib.pyplot as plt
from numpy import  *
from sklearn import svm
from datetime import datetime
from sklearn.cross_validation import cross_val_score
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('data/train.csv')
data1 = pd.read_csv('data/test.csv')
train_data = data.values[0:, 1:]  # 读入全部训练数据
train_data = multiply(train_data, 1.0 / 255.0) # 归一化使得训练速度更快
train_label = data.values[0:, 0]
test_data = data1.values[0:, 0:]  # 测试全部测试个数据
test_data = multiply(test_data, 1.0 / 255.0) # 归一化使得训练速度更快
print 'Data Load Done!'
# return train_data, train_label, test_data
# train_data, train_label, test_data = opencsv()

print shape(train_data),shape(test_data) #训练集有42000个。测试集有28000个 (42000, 784) (28000, 784)

def getncomponent(inputdata):
    pca = PCA()
    pca.fit(inputdata)
    # 累计贡献率,累计方差贡献率
    EV_List = pca.explained_variance_
    EVR_List = []
    for j in range(len(EV_List)):
        EVR_List.append(EV_List[j]/EV_List[0])
    for j in range(len(EVR_List)):
        if(EVR_List[j]<0.10): # 90%以上的都是主要成分
            print 'Recommend %d:' %j
            return j

def modeltest(train_x,train_label,model):
    start = datetime.now()
    metric = cross_val_score(model,train_x,train_label,cv=5,scoring='accuracy').mean() # cv：进行5次交叉验证，train:validate=4:1，通过交叉验证的方法，逐个来验证，可以减少过拟合。如果不懂，请参考：https://blog.csdn.net/qq_36523839/article/details/80707678
    end = datetime.now()
    print 'CV use: %f' %((end-start).seconds)
    print 'Offline Accuracy is %f ' % (metric)
# 方法2超参数搜索
parameters = {'kernel': ('rbf', 'linear'), 'C': (5, 8,10,13,15)}
svr = svm.SVC()
SVM_model = GridSearchCV(svr, parameters)

# kernel:rbf , 惩罚因子10
# SVM_model = svm.SVC(kernel='rbf', C=10)
# components='mle',用mle算法自动算components的大小
pca = PCA(n_components=getncomponent(train_data), whiten=True)
train_x = pca.fit_transform(train_data) # 对train_data进行降维
test_x = pca.transform(test_data) # 用训练好的pca去对test_data进行降维
# modeltest(train_x, train_label, SVM_model) #交叉验证，用来调试svm的参数的c，kernel。

resultname = 'PCA_SVM'
start = datetime.now()
SVM_model.fit(train_x, train_label) # 训练svm
print("The best parameters are %s with a score of %0.2f" % (SVM_model.best_params_,SVM_model.best_score_))
end = datetime.now()
print('train time used:%f' % (end - start).seconds) # 输出训练时间
test_y = SVM_model.predict(test_x)
end = datetime.now()
print('predict time used:%f' % (end - start).seconds) # 输出预测用的时间
pred = [[index + 1, x] for index, x in enumerate(test_y)]
savetxt(resultname + '.csv', pred, delimiter=',', fmt='%d,%d', header='ImageId,Label', comments='')