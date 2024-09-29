# coding:utf-8
import math
import os.path
import shutil

import matplotlib
import pandas as pd
import numpy as np
import openpyxl
import scipy
import math

import sio as sio
import sklearn.utils
from matplotlib import pyplot as plt
from sklearn import svm, preprocessing
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import StratifiedKFold,cross_val_score, cross_validate, train_test_split, cross_val_predict, KFold, \
    GridSearchCV,LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import BaggingRegressor,RandomForestRegressor,AdaBoostRegressor
from sklearn.base import  clone
#from xgboost import XGBRegressor
from sklearn.linear_model import Ridge,LinearRegression,ElasticNet,Lasso
from sklearn.decomposition import PCA
# from brainage_struct.boostrapping import BootstrapWithReq
import seaborn as sns
from sklearn import datasets
import joblib
import statsmodels.formula.api as sm
import scipy.io as sio
def loss(train_score,test_score):
    plt.plot([i for i in range(1, 3)], train_score, 'o-',
             color='r', label='train')
    plt.plot([i for i in range(1, 3)], test_score, 'o-',
             color='g', label='test')

    plt.xlabel('k-fold')
    plt.ylabel('mae-loss')

    plt.legend(loc='best')
    plt.title('pca-5dim')

    plt.show()

def Controllingcovariates(Covariates, X_train, X_test, Covariates_train, Covariates_test):
    Features_Quantity = np.shape(X_train)[1]
    #print(Features_Quantity)
    Covariates_Quantity = np.shape(Covariates)[1]  # Covariates_Quantity = 4 因为有一列subjectkey
    print('Covariates_Quantity',Covariates_Quantity)
    # Controlling covariates from brain data
    df = {}
    for k in np.arange(Covariates_Quantity):
        print('Covariate_-----',Covariates_train[:, k])
        df['Covariate_' + str(k)] = Covariates_train[:, k]  # k+1 避免取到第一列subjectkey

    # Construct formula
    Formula = 'Data ~ Covariate_0'
    for k in np.arange(Covariates_Quantity - 1) + 1:
        Formula = Formula + ' + Covariate_' + str(k)  # Formula = Covariate_0 + Covariate_1 + Covariate_2
   # print(df)
    # Regress covariates from each brain features
    for k in np.arange(Features_Quantity):
        df['Data'] = X_train[:, k]  # 训练集
        # Regressing covariates using training data
        print(df)
        LinModel_Res = sm.ols(formula=Formula,
                              data=df).fit()  # df{'Data':Subjects_Data_train，'Covariate_0':age,'Covariate_1':sex,'Covariate_2':FD}
        # Using residuals replace the training data
        X_train[:, k] = LinModel_Res.resid  # 回归后的结果(新特征，残差)
        # Calculating the residuals of testing data by applying the coeffcients of training data
        Coefficients = LinModel_Res.params
        X_test[:, k] = X_test[:, k] - Coefficients[0]
        for m in np.arange(Covariates_Quantity):  # [0, 1 , 2 ]
            X_test[:, k] = X_test[:, k] - Coefficients[m + 1] * Covariates_test[:, m]#用train的两个变量权重wx+wx=ytest

        print('斜变量回归了～～～～～')
    return X_train, X_test

data = pd.read_csv('/Users/PycharmProjects/pythonProject/abide/714/Z_abide1asdfiq_nclustrrb.csv')
f='fiq_rrb_1'
print(data)

#f='great'
#feature=data.loc[:,f];#to
feature=data.iloc[64:,:28];
print(feature)

feature=pd.DataFrame(feature)
print(feature)
print('#######festure-shape##########',feature.shape)#(109, 208)
r101 = []
mae101 = []
parent_path='/Users/PycharmProjects/pythonProject/NModel/combat-model/'
beh='ADOS_2_RRB'
target = data[beh].values
print('######target-shape###########',target.shape)
print(target)

#conv=data.iloc[:,0:2].values
conv=data.iloc[:,0:2].values
conv=np.array(conv)
print(np.array(conv))

Permutation_Flag=0
for j in range(1,102):
    train_r2_score = []
    test_r2_score = []
    train_mae_score = []
    test_mae_score = []
    predict_label_true = []
    predict_label_test = []

    test_x=[]
    predict_label_true_train = []
    predict_label_pre_train = []

    true_pre = []
    train_pre = []
    feature_share=[]
    kflod = 0

    map_f = {}
    Fold_Corr=[]
    Fold_MAE=[]

    kf = KFold(n_splits=5,shuffle=True)
    #kf=LeaveOneOut()
    l = []
    for X_train,X_test in kf.split(feature):
        kflod=kflod+1;
        x_train,x_test = feature.iloc[X_train,:],feature.iloc[X_test,:]
        y_train,y_test = target[X_train],target[X_test]
        print('no-shuffer---',x_train)
        print('训练集大小：{}'.format(x_train.shape))
        print('测试集大小：{}'.format(x_test.shape))

        #test_x.extend(x_test['zs'].values)

        #Covariates_train, Covariates_test = conv[X_train, :], conv[X_test, :]
        #print('true:x-test:',x_test)
        # # Controlling covariates
        #Xtrain, Xtest = Controllingcovariates(conv, np.array(x_train), np.array(x_test), Covariates_train, Covariates_test)
        Xtrain = x_train
        Xtest = x_test
        print('回归斜变量',Xtest.shape)
        print('回归斜变量', Xtest)
        if Permutation_Flag:
            # If do permutation, the training scores should be permuted, while the testing scores remain
            Subjects_Index_Random = np.arange(len(y_train))
            np.random.shuffle(Subjects_Index_Random)
            y_train = y_train[Subjects_Index_Random]
            print('--shuffer--',y_train)
            if kflod == 1:
                PermutationIndex = {'Fold_0': Subjects_Index_Random}
            else:
                PermutationIndex['Fold_' + str(kflod)] = Subjects_Index_Random



        scaler = preprocessing.MinMaxScaler()
        x_train = scaler.fit_transform(Xtrain)
        x_test = scaler.transform(Xtest)



        C_range = np.exp2(np.arange(16) - 10);
        params = dict(alpha=C_range)
        PLS = PLSRegression()
        # ad = BaggingRegressor(base_estimator=ElasticNet())
        ridge =Ridge()
        elasticNet = ElasticNet()
        svr = svm.SVR(kernel='linear')

        lasso = Lasso()
        linear=LinearRegression()
        clf = GridSearchCV(ridge
                           ,param_grid=params
                           ,cv=5
                           ,verbose=6
                           )
        clf.fit(x_train,y_train)

        cv_results = clf.cv_results_
        print('结果',cv_results)

        # 打印所有参数组合的评价结果

        for mean_score, params in zip(cv_results["mean_test_score"], cv_results["params"]):
            print(np.sqrt(-(mean_score)), params)
            p = str(np.sqrt(-(mean_score)))+'_'+str(params)
            l.append(p)
        df = pd.DataFrame(l)
        path = '/Users/PycharmProjects/pythonProject/abide/714/rrb_0/'+str(j)+'csv'
        df.to_csv(path)
        model_best = clf.best_estimator_
        #model_best.fit(x_train,y_train)
        print('-----model_best---------',model_best)
        # model='101_'+str(j)+'_kflod_'+str(kflod)+'pkl'
        # final_model_path=parent_path+model
        # joblib.dump(model_best,final_model_path)

        pred_Ytrain = model_best.predict(x_train)
        pred_Ytest = model_best.predict(x_test)
        print("------_xtest--------", x_test.shape)
        print("------pred_Ytest--------",pred_Ytest)
        print("------true--------", y_test)

        # pred_Ytest = pred_Ytest[:,0]#plsr
        # pred_Ytrain = pred_Ytrain[:, 0]  # plsr
        Fold_J_Corr = np.corrcoef(pred_Ytest, y_test)
        print('------corr--------',Fold_J_Corr)
        Fold_J_Corr = Fold_J_Corr[0, 1]
        Fold_Corr.append(Fold_J_Corr)
        Fold_J_MAE = np.mean(np.abs(np.subtract(pred_Ytest, y_test)))
        Fold_MAE.append(Fold_J_MAE)



        predict_label_true.extend(y_test)
        predict_label_test.extend(pred_Ytest)

        predict_label_true_train.extend(y_train)
        predict_label_pre_train.extend(pred_Ytrain)

        train_r2_score.append(r2_score(y_train, pred_Ytrain))
        test_r2_score.append(r2_score(y_test, pred_Ytest))

        train_mae_score.append(mean_absolute_error(y_train, pred_Ytrain))
        test_mae_score.append(mean_absolute_error(y_test, pred_Ytest))
        print('===weight=',model_best.coef_.shape)#(1,210)
        Weight = model_best.coef_ / np.sqrt(np.sum(model_best.coef_ ** 2))#ridge

        print('weight',Weight)

        feature_share.append(Weight)

    # true_pre.append(test_x)
    # print(np.array(test_x))#toc保存x值
    true_pre.append(predict_label_true)
    true_pre.append(predict_label_test )
    print(np.array(true_pre).shape)
    true_pre = np.array(true_pre).transpose(1,0)
    df = pd.DataFrame(true_pre)

    result_fold='/Users/PycharmProjects/pythonProject/abide/714/rrb_0/dataset_'+f+beh

    if os.path.exists(result_fold):
        print('exit')
    else:
        os.mkdir(result_fold)
    rp = '/Users/PycharmProjects/pythonProject/abide/714/rrb_0/dataset_' + f + beh + '/pre_score_' + str(j) + '.csv'
    df.to_csv(rp)


    train_pre.append(predict_label_true_train)
    train_pre.append(predict_label_pre_train )
    train_pre = np.array(train_pre).transpose(1,0)

    print('第{}模型'.format(j))
    print('训练集的r2：{}'.format(np.array(train_r2_score).mean()))
    print('训练集的mae：{}'.format(np.array(train_mae_score).mean()))
    print('测试集的r2：{}'.format(np.array(test_r2_score).mean()))
    print('测试集的mae：{}'.format(np.array(test_mae_score).mean()))


    Fold_Corr = [0 if np.isnan(x) else x for x in Fold_Corr]
    print('corr--shape:',np.array(Fold_Corr).shape)
    Mean_Corr = np.mean(Fold_Corr)
    Mean_MAE = np.mean(Fold_MAE)
    Res_NFold = {'Mean_Corr': Mean_Corr, 'Mean_MAE': Mean_MAE};
    print('测试集Meancorr:',Mean_Corr)
    print('测试集Mean_MAE',Mean_MAE)

    print('fature-shape:',np.array(feature_share).shape)

    Mean_weight = np.mean(feature_share,axis=0)

    ds = pd.DataFrame(Mean_weight)
    rp='/Users/PycharmProjects/pythonProject/abide/714/rrb_0/dataset_'+f+beh+'/feature-weight'+str(j)+'.csv'
    ds.to_csv(rp)

    r101.append(Mean_Corr)
    mae101.append(Mean_MAE)
print(len(r101))
print(r101)
print(len(mae101))
if Permutation_Flag==0:
    print('median-r:',np.median(np.array(r101)))
    m=np.median(np.array(r101));
    print('median-mae:', np.median(np.array(mae101)))
    mae=np.median(np.array(mae101));
    path='/Users/PycharmProjects/pythonProject/abide/714/rrb_0/dataset_'+f+beh+'/1media-r.txt'
    model_index=np.where(np.array(r101)==m)[0][0]+1

    print(model_index)
    print('第{}模型：'.format(model_index))
    with open(path,'w') as f:
        f.write('media-r:'+str(m))
        f.write('\n')
        f.write('media-mae:' + str(mae))
        f.write('\n')
        f.write('model：'+str(model_index))

    # print(np.where(np.array(r101)==m))



