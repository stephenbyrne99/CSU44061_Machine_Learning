# id:25-50--25-0 - data set 1
# id:25-25-25-0  - data set 2

# Notes - need to use cross validation and k folds

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.dummy import DummyClassifier

def split(X1,X2,y):
    """ Splits data of two features (X1,X2) into positive and negative data point arrays 
        using result of data (y). Returns two arrays (positive and negative data point pairs)

    """

    X1positive = []
    X2positive = []

    X1negative = []
    X2negative = []

    # iterate over both 
    for (x1i, x2i, yi) in zip(X1, X2, y):
       
        if(yi == 1):
             # if positive add to positive array
            X1positive.append(x1i)
            X2positive.append(x2i)
        else:
             # if negative add to negative array
            X1negative.append(x1i)
            X2negative.append(x2i)

    Xpositive = np.column_stack((X1positive,X2positive))
    Xnegative = np.column_stack((X1negative,X2negative))

    return Xpositive, Xnegative

def create_test_space(size):
    # create test space
    X_test_space = []
    grid = np.linspace(size * -1,size)
    for i in grid:
        for j in grid:
            X_test_space.append([i,j])
    X_test_space = np.array(X_test_space)
    return X_test_space

def plot_splitted_data(positive,negative):
    plt.title('Visualisation of training data')
    plt.xlabel('first training feature')
    plt.ylabel('second training feature')
    plt.scatter(positive[:,0],positive[:,1], label="Positive Train",color='g',marker='+')
    plt.scatter(negative[:,0],negative[:,1], label="Negative Train",color='b')
    plt.legend()
    plt.show()


def plot_correct_incorrect_graph(Xcorrect,Xincorrect,title):
    plt.title(title)
    plt.xlabel('first training feature')
    plt.ylabel('second training feature')
    plt.scatter(Xcorrect[:,0],Xcorrect[:,1], label="Correct predictions",color='g')
    plt.scatter(Xincorrect[:,0],Xincorrect[:,1], label="Incorrect Predictions",color='r')
    plt.legend()
    plt.show()

def split_by_correct_prediction(X1,X2,y,y_pred):
    """ Splits data by correct / incorrect prediction

    """

    X1correct = []
    X2correct = []

    X1incorrect = []
    X2incorrect = []

    # iterate over both 
    for (x1i, x2i, yi, yi_pred) in zip(X1, X2, y, y_pred):

        if(yi == yi_pred):
             # if positive add to positive array
            X1correct.append(x1i)
            X2correct.append(x2i)
        else:
             # if negative add to positive array
            X1incorrect.append(x1i)
            X2incorrect.append(x2i)

    Xcorrect = np.column_stack((X1correct,X2correct))
    Xincorrect = np.column_stack((X1incorrect,X2incorrect))

    return Xcorrect, Xincorrect

def plot_training_and_test_data_graph(positive_train,negative_train,positive_test,negative_test,title):
    plt.title(title)
    plt.xlabel('first training feature')
    plt.ylabel('second training feature')
    plt.scatter(positive_train[:,0],positive_train[:,1], label="Positive Training data",color='g',marker='+')
    plt.scatter(negative_train[:,0],negative_train[:,1], label="Negative Training data",color='b')
    plt.scatter(positive_test[:,0],positive_test[:,1], label="Positive Test",color='r',marker='+')
    plt.scatter(negative_test[:,0],negative_test[:,1], label="Negative Test",color='r',marker='o')
    plt.legend()
    plt.show()

def get_most_common_case(y):
    sum = np.sum(y) 
    if sum > 0:
        return 1
    if sum < 0:
        return -1
    return 0

def main():
     # adjust graph values so legibly
    plt.rc('font', size=16)
    plt.rcParams['figure.constrained_layout.use'] = True

    # Read in data - change 1 to 2 for different data set
    df = pd.read_csv('week4-2.csv')
    print(df.head())
    X1=df.iloc[: ,0]
    X2=df.iloc[: ,1]
    X=np.column_stack(( X1, X2 ))
    y=df.iloc[: , 2]

    # split into positive and negative arrays so can visualise
    positive, negative = split(X1,X2,y)

    # plot data
    plot_splitted_data(positive,negative)

    #
    #
    #               LOGISTIC ################################
    #
    #
      
    #VISUAL GRAPHS NO K FOLDS
    # polynomial_features = [1,2,3,4,5]
    # for degree in polynomial_features:
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    #     c = 1
    #     poly_transform = PolynomialFeatures(degree=degree)
    #     model_logistic_c_1 = LogisticRegression(penalty="l2",C=c)
    #     model_logistic_c_1 .fit(poly_transform.fit_transform(X_train),y_train)
    #     print('Lasso C=',c,'Coef : ' , model_logistic_c_1 .coef_)
    #     print('Lasso C=',c,' Intercept : ' , model_logistic_c_1.intercept_)

    #     X_test_space = create_test_space(5)
    #     y_pred_i = model_logistic_c_1.predict(poly_transform.fit_transform(X_test))
    #     positive_test,negative_test = split(X_test[:,0],X_test[:,1],y_pred_i)
    #     positive_train,negative_train = split(X_train[:,0],X_train[:,1],y_train)
    #     title = 'Test Results Logistic Poly Degree=' + str(degree)
    #     plot_training_and_test_data_graph(positive_train,negative_train,positive_test,negative_test,title)
    #     Xcorrect, Xincorrect = split_by_correct_prediction(X_test[:,0],X_test[:,1],y_test, y_pred_i)
    #     plot_correct_incorrect_graph(Xcorrect, Xincorrect, title)

    
    # K FOLDS TO GET STD/MEAN W/ POLYNOMIALS
    polynomial_features = [1,2,3,4,5]
    std_devs = []
    means = []
    for degree in polynomial_features:
        kf = KFold(n_splits=5)
        c = 1
        mse_estimates = []
        for train, test in kf.split(X):
            poly_transform = PolynomialFeatures(degree=degree)
            model_logistic_c_1 = LogisticRegression(penalty="l2",C=c)
            model_logistic_c_1 .fit(poly_transform.fit_transform(X[train]),y[train])

            X_test_space = create_test_space(1)
            y_pred_i = model_logistic_c_1.predict(poly_transform.fit_transform(X[test]))
            mse_estimates.append(mean_squared_error(y_pred_i, y[test]))

        means.append(np.mean(mse_estimates))
        std_devs.append(np.std(mse_estimates))

    print(means)
    print(std_devs)

    plt.errorbar(polynomial_features,means,yerr=std_devs,linewidth=3,capsize=5)
    plt.title('Logistic Regression for Different polynomial features')
    plt.xlabel('Polynomial Features'); 
    plt.ylabel('Mean/STD dev value')
    plt.show()

    #NEED TO ADD IN KFOLDS VALIDATION AND THEN GRAPH RESULTS
    # c_values = [0.001,0.01,0.1,1,10,100,1000]
    # for ci in c_values:
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    #     model_logistic_c_i = LogisticRegression(penalty="l2",C=ci)
    #     model_logistic_c_i .fit(poly_transform.fit_transform(X_train),y_train)
    #     print('Lasso C=',ci,'Coef : ' , model_logistic_c_i .coef_)
    #     print('Lasso C=',ci,' Intercept : ' , model_logistic_c_i .intercept_)

    #     X_test_space = create_test_space(5)
    #     y_pred_i = model_logistic_c_i.predict(poly_transform.fit_transform(X_test))
    #     positive_test,negative_test = split(X_test[:,0],X_test[:,1],y_pred_i)
    #     positive_train,negative_train = split(X_train[:,0],X_train[:,1],y_train)
    #     title = 'Test Results Logistic C=' + str(ci)
    #     plot_training_and_test_data_graph(positive_train,negative_train,positive_test,negative_test,title)
    #     Xcorrect, Xincorrect = split_by_correct_prediction(X_test[:,0],X_test[:,1],y_test, y_pred_i)

    #     plot_correct_incorrect_graph(Xcorrect, Xincorrect, title)

    # K FOLDS TO GET STD/MEAN W/ C Values
    c_values = [0.001,0.01,0.1,1,10,100,1000]
    std_devs = []
    means = []
    for ci in c_values:
        kf = KFold(n_splits=5)
        mse_estimates = []
        for train, test in kf.split(X):
            poly_transform = PolynomialFeatures(degree=2)
            model_logistic_c_1 = LogisticRegression(penalty="l2",C=ci)
            model_logistic_c_1 .fit(poly_transform.fit_transform(X[train]),y[train])
            y_pred_i = model_logistic_c_1.predict(poly_transform.fit_transform(X[test]))
            mse_estimates.append(mean_squared_error(y_pred_i, y[test]))

        means.append(np.mean(mse_estimates))
        std_devs.append(np.std(mse_estimates))

    print(means)
    print(std_devs)

    plt.errorbar(np.log10(c_values),means,yerr=std_devs,linewidth=3,capsize=5)
    plt.title('Logistic Regression for Different Values of C')
    plt.xlabel('Log(C) values'); 
    plt.ylabel('Mean/STD dev value')
    plt.show()

    #
    #
    #               kNN ################################
    #
    #

    # K FOLDS TO show polynimial doesnt matter
    polynomial_features = [1,2,3,4,5]
    std_devs = []
    means = []
    for degree in polynomial_features:
        n=5
        kf = KFold(n_splits=5)
        mse_estimates = []
        for train, test in kf.split(X):
            model_kNN_ni = KNeighborsClassifier(n_neighbors=n,weights="uniform").fit(poly_transform.fit_transform(X[train]),y[train])
            y_pred_i = model_kNN_ni.predict(poly_transform.fit_transform(X[test]))
            mse_estimates.append(mean_squared_error(y_pred_i, y[test]))

        means.append(np.mean(mse_estimates))
        std_devs.append(np.std(mse_estimates))

    print(means)
    print(std_devs)

    plt.errorbar(polynomial_features,means,yerr=std_devs,linewidth=3,capsize=5)
    plt.title('kNN for Different Poly Features')
    plt.xlabel('Poly features'); 
    plt.ylabel('Mean/STD dev value')
    plt.show()

    # k_neighbours = [1,2,3,4,5]
    # for ni in k_neighbours:
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    #     model_kNN_ni = KNeighborsClassifier(n_neighbors=ni,weights="uniform").fit(X[train],y[train])
    #     y_pred_i = model_kNN_ni.predict(X[test])
    #     positive_test,negative_test = split(X_test[:,0],X_test[:,1],y_pred_i)
    #     positive_train,negative_train = split(X_train[:,0],X_train[:,1],y_train)
    #     title = 'Test Results kNN Neighbours N=' + str(ni)
    #     plot_training_and_test_data_graph(positive_train,negative_train,positive_test,negative_test,title)
    #     Xcorrect, Xincorrect = split_by_correct_prediction(X_test[:,0],X_test[:,1],y_test, y_pred_i)
    #     plot_correct_incorrect_graph(Xcorrect, Xincorrect, title)
    


    # K FOLDS TO GET BEST NEIGHBOURS
    neighbours = [1,2,3,4,5,6,7,8,9,10,25,50,100]
    std_devs = []
    means = []
    for n in neighbours:
        kf = KFold(n_splits=5)
        mse_estimates = []
        for train, test in kf.split(X):
            model_kNN_ni = KNeighborsClassifier(n_neighbors=n,weights="uniform").fit(X[train],y[train])
            y_pred_i = model_kNN_ni.predict(X[test])
            mse_estimates.append(mean_squared_error(y_pred_i, y[test]))

        means.append(np.mean(mse_estimates))
        std_devs.append(np.std(mse_estimates))

    print(means)
    print(std_devs)

    plt.errorbar(neighbours,means,yerr=std_devs,linewidth=3,capsize=5)
    plt.title('kNN for Different Neighbours')
    plt.xlabel('Neighbours'); 
    plt.ylabel('Mean/STD dev value')
    plt.show()

    # Using best models calculate confusion matrices 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    model_kNN_5 = KNeighborsClassifier(n_neighbors=50,weights="uniform").fit(X_train,y_train)
    y_pred_kNN = model_kNN_5.predict(X_test)
    Xcorrect, Xincorrect = split_by_correct_prediction(X_test[:,0],X_test[:,1],y_test, y_pred_kNN)
    plot_correct_incorrect_graph(Xcorrect, Xincorrect, 'kNN k=50')
    print('CONFUSION MATRIX FOR KNN N=50')
    print(confusion_matrix(y_test, y_pred_kNN))

    poly_transform = PolynomialFeatures(degree=2)
    model_logistic_c_1 = LogisticRegression(penalty="l2",C=1)
    model_logistic_c_1.fit(poly_transform.fit_transform(X_train),y_train)
    y_pred_logistic = model_logistic_c_1.predict(poly_transform.fit_transform(X_test))
    print('CONFUSION MATRIX FOR LOGISTIC C=1 W/ 2 FEATURES')
    print(confusion_matrix(y_test, y_pred_logistic))

    # Compare against baseline that always predicts most common case
    most_common_case = get_most_common_case(y)

    print('most common case : ', most_common_case)

    # fill with most common case
    baseline_prediction = np.ones(len(y_test))

    # Show the Confusion Matrix
    print(confusion_matrix(y_test, baseline_prediction))

    # use sklearn for predicting at random
    dummy_clf = DummyClassifier(strategy="uniform")
    dummy_clf.fit(X_train, y_train)
    y_pred_dummy = dummy_clf.predict(X_test)
    print('Dummy classifier ', dummy_clf.score(X_test, y_test)) # retruns mean accuracy 
    true_positive = dummy_clf.score(X_test, y_test)
    false_positive = 1 - true_positive
    print('TP' , true_positive)
    print('FP' , false_positive)


    print('CONFUSION MATRIX FOR DUMMY PRED RANDOM')
    print(confusion_matrix(y_test, y_pred_dummy))


    
    # Plot ROC curves for trained models
    fpr,tpr, threshold = roc_curve(y_test,model_logistic_c_1.decision_function(poly_transform.fit_transform(X_test)))
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.plot(1,1, label='Most Frequent',marker='o')
    plt.plot(0.5,0.5, label='Random',marker='o')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for logistic')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.show()

    y_score = model_kNN_5.predict_proba(X_test)
    fpr,tpr, threshold = roc_curve(y_test,y_score[:,1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.plot(1,1, label='Most Frequent',marker='o')
    plt.plot(0.5,0.5, label='Random',marker='o')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC Curve for kNN')
    plt.show()

    fpr,tpr, threshold = roc_curve(y_test,model_logistic_c_1.decision_function(poly_transform.fit_transform(X_test)))
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label = 'AUC Logistic = %0.2f' % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for logistic')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')

    y_score = model_kNN_5.predict_proba(X_test)
    fpr,tpr, threshold = roc_curve(y_test,y_score[:,1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'r', label = 'AUC kNN= %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'g--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.plot(1,1, label='Most Frequent',marker='o')
    plt.plot(0.5,0.5, label='Random',marker='o')
    plt.title('Both ROC curves')
    plt.show()


    # Plot points on the ROC plot corresponding to baseline classifiers

    # also - AUC on random = 0.5


    # Evaluate the performance between the two


if __name__ == "__main__":
    main()
