# id:24-24--24 

import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import math   
from sklearn.dummy import DummyClassifier

def create_test_space(size):
    # create test space
    X_test_space = []
    grid = np.linspace(size * -1,size)
    for i in grid:
        for j in grid:
            X_test_space.append([i,j])
    X_test_space = np.array(X_test_space)
    return X_test_space

def graph_3d_prediction(X,y,y_pred,title):
    X_test_space = create_test_space(2)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    surface = ax.plot_trisurf(X_test_space[:,0], X_test_space[:,1], y_pred, cmap='viridis', edgecolor='none')
    ax.scatter(X[:,0],X[:,1],y,c='blue',label="Training Data")
    ax.set_xlabel('first feature')
    ax.set_ylabel('second feature')
    ax.set_zlabel('target')
    fig.colorbar(surface, shrink=0.5, aspect=5, label='Predictions')
    plt.title(title)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),shadow=True, ncol=2)
    plt.show(fig)


def main_i():

    """
        Part (i)
    """

    #####################################################
    #                                                   #
    #                  Question A                       #
    #                                                   #
    #####################################################

    # adjust graph values so legibly
    plt.rc('font', size=16)
    plt.rcParams['figure.constrained_layout.use'] = True

    # Read in data    
    df = pd.read_csv('week3.csv')
    print(df.head())
    X1=df.iloc[: ,0]
    X2=df.iloc[: ,1]
    X=np.column_stack(( X1, X2 ))
    y=df.iloc[: , 2]

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(X[:,0],X[:,1],y,c='blue')
    ax.set_xlabel('first feature')
    ax.set_ylabel('second feature')
    ax.set_zlabel('target')

    plt.title('3D Scatter Plot of Data')
    plt.show(fig)

    #####################################################
    #                                                   #
    #                  Question B & C                   #
    #                                                   #
    #####################################################

    print(df.shape)

    poly_transform = PolynomialFeatures(degree=5)  
    # chang4 qll or below into loops
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # all lassos

    # C = [1,2,5,10,1000]
    c_values = [1,2,10,1000]
    for ci in c_values:
        ai = 1/(2*ci)
        model_lasso_c_i = Lasso(alpha=ai)
        model_lasso_c_i.fit(poly_transform.fit_transform(X_train),y_train)
        print('Lasso C=',ci,'Coef : ' , model_lasso_c_i.coef_)
        print('Lasso C=',ci,' Intercept : ' , model_lasso_c_i.intercept_)

        X_test_space = create_test_space(5)
        y_pred_i = model_lasso_c_i.predict(poly_transform.fit_transform(X_test_space))
        title = 'Test Results Lasso C=' + str(ci)
        graph_3d_prediction(X_train,y_train,y_pred_i,title)


    #####################################################
    #                                                   #
    #                  Question D                       #
    #                                                   #
    #####################################################

    # EXPLAIN PARAMETERS

    #####################################################
    #                                                   #
    #                  Question E                       #
    #                                                   #
    #####################################################


    # # C = [1,2,10,1000]
    c_values = [0.0000001,0.00001,0.001,1]
    for ci in c_values:
        ai = 1/(2*ci)
        model_ridge_c_i = Ridge(alpha=ai)
        model_ridge_c_i.fit(poly_transform.fit_transform(X_train),y_train)
        print('Ridge C=',ci,'Coef : ' , model_ridge_c_i.coef_)
        print('Ridge C=',ci,' Intercept : ' , model_ridge_c_i.intercept_)

        X_test_space = create_test_space(4)
        y_pred_i = model_ridge_c_i.predict(poly_transform.fit_transform(X_test_space))
        title = 'Test Results Ridge C=' + str(ci)
        graph_3d_prediction(X_train,y_train,y_pred_i,title)
    
    """
        Part (ii)
    """

    #####################################################
    #                                                   #
    #                  Question A                       #
    #                                                   #
    #####################################################

    model_lasso_c_1 = Lasso(alpha=0.5)
    folds = [2,5,10,25,40,100]
    variances = []
    means = []
    for fold in folds:
        kf = KFold(n_splits=fold)
        mse_estimates = []
        for train, test in kf.split(X):
            model_lasso_c_1.fit(poly_transform.fit_transform(X[train]),y[train])
            X_test_space = create_test_space(4)
            y_pred_1 = model_lasso_c_1.predict(poly_transform.fit_transform(X[test]))
            mse_estimates.append(mean_squared_error(y_pred_1, y[test]))
        

        means.append(np.mean(mse_estimates))
        variances.append(np.var(mse_estimates))


    print(variances)
    print(means)

    plt.errorbar(folds,means,yerr=np.sqrt(variances),linewidth=3,capsize=5)
    plt.title('Lasso c=1 for multiple folds')
    plt.xlabel('Number of folds'); 
    plt.ylabel('Mean/StdDev Value')
    plt.show()


    #####################################################
    #                                                   #
    #                  Question B                       #
    #                                                   #
    #####################################################

    C_values = [1,2,10,1000]
    std_devs = []
    means = []
    for Ci in C_values:
        kf = KFold(n_splits=5)
        alpha = 1/(2*Ci)
        model_lasso_c_i = Lasso(alpha=alpha)
        mse_estimates = []
        for train, test in kf.split(X):
            model_lasso_c_i .fit(poly_transform.fit_transform(X[train]),y[train])
            X_test_space = create_test_space(4)
            y_pred_1 = model_lasso_c_i.predict(poly_transform.fit_transform(X[test]))
            mse_estimates.append(mean_squared_error(y_pred_1, y[test]))

        means.append(np.mean(mse_estimates))
        std_devs.append(np.std(mse_estimates))


    print(means)
    print(std_devs)

    plt.errorbar(C_values,means,yerr=std_devs,linewidth=3,capsize=5)
    plt.title('5 Fold Lasso for multiple C values')
    plt.xlabel('C Values'); 
    plt.ylabel('Mean/STD dev value')
    plt.show()

    #####################################################
    #                                                   #
    #                  Question C                       #
    #                                                   #
    #####################################################
    
    #####################################################
    #                                                   #
    #                  Question D                       #
    #                                                   #
    #####################################################

    model_ridge_c_1 = Lasso(alpha=0.5)
    folds = [2,5,10,25,40,100]
    variances = []
    means = []
    for fold in folds:
        kf = KFold(n_splits=fold)
        mse_estimates = []
        for train, test in kf.split(X):
            model_ridge_c_1.fit(poly_transform.fit_transform(X[train]),y[train])
            X_test_space = create_test_space(4)
            y_pred_1 = model_ridge_c_1.predict(poly_transform.fit_transform(X[test]))
            mse_estimates.append(mean_squared_error(y_pred_1, y[test]))
        

        means.append(np.mean(mse_estimates))
        variances.append(np.var(mse_estimates))


    print(variances)
    print(means)

    plt.errorbar(folds,means,yerr=variances,linewidth=3,capsize=5)
    plt.xlabel('folds'); 
    plt.ylabel('values');
    plt.title('Ridge C=1 for different folds')
    plt.show()


    C_values = [0.0000001,0.00001,0.001,1,10,1000]
    std_devs = []
    means = []
    for Ci in C_values:
        kf = KFold(n_splits=5)
        alpha = 1/(2*Ci)
        model_ridge_c_i = Ridge(alpha=alpha)
        mse_estimates = []
        for train, test in kf.split(X):
            model_ridge_c_i .fit(poly_transform.fit_transform(X[train]),y[train])
            X_test_space = create_test_space(4)
            y_pred_1 = model_ridge_c_i.predict(poly_transform.fit_transform(X[test]))
            mse_estimates.append(mean_squared_error(y_pred_1, y[test]))

        means.append(np.mean(mse_estimates))
        std_devs.append(np.std(mse_estimates))


    print(means)
    print(std_devs)

    plt.errorbar(np.log10(C_values),means,yerr=std_devs,linewidth=3,capsize=5)
    plt.xlabel('Log10(C_Values)'); 
    plt.ylabel('values');
    plt.title('Ridge for multiple c values')
    plt.show()


if __name__ == "__main__":
    main_i()
