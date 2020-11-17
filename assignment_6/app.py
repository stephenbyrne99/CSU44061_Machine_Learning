# id:25--125--125 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge

def create_test_space(size):
    # create test space
    X_test_space = []
    grid = np.linspace(size * -1,size)
    for i in grid:
        for j in grid:
            X_test_space.append([i,j])
    X_test_space = np.array(X_test_space)
    return X_test_space

def gaussian_kernel(distances):
    weights = np.exp(-25*(distances**2))
    return weights/np.sum(weights)

def main_i():
    # adjust graph values so legibly
    plt.rc('font', size=16)
    plt.rcParams['figure.constrained_layout.use'] = True


    #dummy_data = np.array([[-1,0],[0,1],[1,0]])

    # X_train = np.array([-1,0,1]).reshape(-1,1)
    # y_train = np.array([0,1,0])

    # model = KNeighborsRegressor(n_neighbors=3,weights=gaussian_kernel).fit(dummy_data[:,0],dummy_data[:,1])

    # y_pred = model.predict(dummy_data[:,0])

    # plt.scatter(dummy_data[:,0],dummy_data[:,1], color="red", marker="+")
    # plt.plot(dummy_data[:,0], y_pred, color="green")
    # plt.xlabel("input x"); 
    # plt.ylabel("output y")
    # plt.legend("kNN")
    # plt.show()

    m = 3
    Xtrain = np.linspace(-1.0,1.0,num=m)
    ytrain = np.array([0,1,0])
    Xtrain = Xtrain.reshape(-1,1)
    # C=10;
    # model = KernelRidge(alpha=1.0/C, kernel="rbf", gamma=10).fit(Xtrain, ytrain)
    Xtest=np.linspace(-3,3,num=1000).reshape(-1,1)
    # ypred = model.predict(Xtest)
    model2 = KNeighborsRegressor(n_neighbors=m,weights=gaussian_kernel).fit(Xtrain, ytrain)
    ypred2 = model2.predict(Xtest)
    plt.scatter(Xtrain, ytrain, color="red", marker="+")
    # plt.plot(Xtest, ypred, color="green")
    plt.plot(Xtest, ypred2, color="blue")
    plt.xlabel("input x"); plt.ylabel("output y")
   # plt.legend(["Kernel Ridge Regression","kNN","train"])
    plt.show()

    

 

if __name__ == "__main__":
    main_i()
    # main_ii()