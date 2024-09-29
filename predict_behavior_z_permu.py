# coding:utf-8
import math
import os.path
import shutil
import matplotlib
import pandas as pd
import numpy as np
import scipy
import math
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

def Controllingcovariates(Covariates, X_train, X_test, Covariates_train, Covariates_test):
    Features_Quantity = np.shape(X_train)[1]
    #print(Features_Quantity)
    Covariates_Quantity = np.shape(Covariates)[1] 
    # Controlling covariates from brain data
    df = {}
    for k in np.arange(Covariates_Quantity):
        df['Covariate_' + str(k)] = Covariates_train[:, k]  

    # Construct formula
    Formula = 'Data ~ Covariate_0'
    for k in np.arange(Covariates_Quantity - 1) + 1:
        Formula = Formula + ' + Covariate_' + str(k)  
   # print(df)
    # Regress covariates from each brain features
    for k in np.arange(Features_Quantity):
        df['Data'] = X_train[:, k]  # 训练集
        # Regressing covariates using training data
        #print(df)
        LinModel_Res = sm.ols(formula=Formula,
                              data=df).fit() 
        X_train[:, k] = LinModel_Res.resid  
        
        Coefficients = LinModel_Res.params
        X_test[:, k] = X_test[:, k] - Coefficients[0]
        for m in np.arange(Covariates_Quantity):  # [0, 1 , 2 ]
            X_test[:, k] = X_test[:, k] - Coefficients[m + 1] * Covariates_test[:, m]#用train的两个变量权重wx+w2x2=ytest

        print('斜变量回归了～～～～～')
    return X_train, X_test

data = pd.read_csv('/nmodel/abide/Z_abide1asd_countfiqcc.csv')
f='count_fiq_cc'
bev='outliers_counts'
feature=data.loc[:,bev];
#feature=data.iloc[71:,:28]
feature=pd.DataFrame(feature)
print(feature)
print('#######festure-shape##########',feature.shape)#(109, 208)
r101 = []
mae101 = []
parent_path='/nmodel/model/'
beh='ADOS_G_COMM'
print(beh)
target = data[beh].values
print('######target-shape###########',target.shape)
#conv=data.iloc[:,0:2].values
conv=data.iloc[:,0:2].values
conv=np.array(conv)
print(np.array(conv).shape)
Permutation_Flag=1
for j in range(1,1001):
    train_r2_score = []
    test_r2_score = []
    train_mae_score = []
    test_mae_score = []
    predict_label_true = []
    predict_label_test = []

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
    
    for X_train,X_test in kf.split(feature):
        kflod=kflod+1;
        x_train,x_test = feature.iloc[X_train,:],feature.iloc[X_test,:]
        y_train,y_test = target[X_train],target[X_test]
        print('no-shuffer---',y_train)
        print('训练集大小：{}'.format(x_train.shape))
        print('测试集大小：{}'.format(x_test.shape))

        #Covariates_train, Covariates_test = conv[X_train, :], conv[X_test, :]

        # # Controlling covariates
        #Xtrain, Xtest = Controllingcovariates(conv, np.array(x_train), np.array(x_test), Covariates_train, Covariates_test)
        Xtrain = x_train
        Xtest = x_test
        print('no-shuffer--:',y_train)
        print('Xtest--shape:',Xtest.shape)
        print('Xtest--shape:',Xtest)
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


        # print(x_train)
        scaler = preprocessing.MinMaxScaler()
        x_train = scaler.fit_transform(Xtrain)
        x_test = scaler.transform(Xtest)

        
        my_func = make_scorer(my_scorer, greater_is_better=True)
        C_range = np.exp2(np.arange(16) - 10);
  
        params = dict(alpha=C_range)
        
        ridge =Ridge()
        
        clf = GridSearchCV(ridge
                           , param_grid=params
                           , cv=5
                           
                           )
        clf.fit(x_train,y_train)
        model_best = clf.best_estimator_
        print('-----model_best---------',model_best)

        

        pred_Ytrain = model_best.predict(x_train)
        pred_Ytest = model_best.predict(x_test)
        print("------pred_Ytest--------",pred_Ytest)
        print("------true--------", y_test)

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
       
        Weight = model_best.coef_ / np.sqrt(np.sum(model_best.coef_ ** 2))#
        
        feature_share.append(Weight)


    true_pre.append(predict_label_true)
    true_pre.append(predict_label_test )

    true_pre = np.array(true_pre).transpose(1,0)
    df = pd.DataFrame(true_pre)

    result_fold='/nmodel/combat/dataset_permu_'+f+'_'+beh
    print(result_fold)
    if os.path.exists(result_fold):
        print('exit')
    else:
        os.mkdir(result_fold)
    rp = './combat/dataset_permu_' + f + '_'+beh + '/pre_score_' + str(j) + '.csv'
    df.to_csv(rp)


    train_pre.append(predict_label_true_train)
    train_pre.append(predict_label_pre_train )
    train_pre = np.array(train_pre).transpose(1,0)
    df = pd.DataFrame(train_pre)
    rp='./combat/dataset_permu_'+f+'_'+beh+'/train_pre_score_'+str(j)+'.csv'
    df.to_csv(rp)


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
    

    print('weight-shape--:',np.array(feature_share).shape)

    Mean_weight = np.mean(feature_share,axis=0)

    ds = pd.DataFrame(Mean_weight)
    rp='./combat/dataset_permu_'+f+'_'+beh+'/feature-weight'+str(j)+'.csv'
    ds.to_csv(rp)

    #loss(train_mae_score,test_mae_score)

    r101.append(Mean_Corr)
    mae101.append(Mean_MAE)
print(len(r101))
print(r101)
print(len(mae101))
if Permutation_Flag:

    r=0.13073204356317675
    r101=np.array(r101)
    count=np.sum(r101>=r)
    p=(1+count)/1001.0;
    print((1+count)/1001.0)

    m=1.363986909807421
    mae101=np.array(mae101)
    countm=np.sum(mae101<=m)
    pm=(1+countm)/1001.0
    print((1+countm)/1001.0)


    path = './combat/dataset_permu_'+f+'_'+beh + '/1permu-p.txt'
    with open(path,'w') as f:
        f.write('media-r:'+str(r))
        f.write('\n')
        f.write('p：'+str(p))
        f.write('\n')
        f.write('media-mae:'+str(m))
        f.write('\n')
        f.write('pmae：'+str(pm))
