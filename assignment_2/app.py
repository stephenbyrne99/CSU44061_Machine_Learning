# id:2-4-2 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC


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
    



def main_a():
    """ Week 2 Machine Learning Assignment Q A

                Logistic Regression

    """

#####################################################
#                                                   #
#                  Question A                       #
#                                                   #
#####################################################

    # adjust graph values so legibly
    plt.rc('font', size=14)
    plt.rcParams['figure.constrained_layout.use'] = True

    # Read in data    
    df = pd.read_csv('week2.csv')
    print(df.head())
    X1=df.iloc[: ,0]
    X2=df.iloc[: ,1]
    X=np.column_stack(( X1, X2 ))
    y=df.iloc[: , 2]

    # split into positive and negative arrays so can visualise
    positive, negative = split(X1,X2,y)

    # plot data
    plot_splitted_data(positive,negative)


# logistic regression

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # split to visualise training data
    positive_train, negative_train = split(X_train[:,0],X_train[:,1],y_train)

    # # plot splitted data
    # plot_splitted_data(positive_train, negative_train)

    #train model
    model = LogisticRegression()
    model.fit(X, y)
    y_pred = model.predict(X_test)

    # all atributes
    classes = model.classes_
    coef = model.coef_
    intercept = model.intercept_

    print('classes : ,' ,classes)
    print('coef : ,' ,coef)
    print('intercept : ,' ,intercept)
    print(y_pred)


    # Show the Confusion Matrix
    print(confusion_matrix(y_test, y_pred))

    # get model paramete values to use when plot decision boundary  # from online article - https://scipython.com/blog/plotting-the-decision-boundary-of-a-logistic-regression-model/

    b = intercept[0]
    w1, w2 = coef.T


    # split test data by predictions
    positive_test,negative_test = split(X_test[:,0],X_test[:,1],y_pred)

    # add decision boundary
    add_decision_boundary_to_plot(b,w1,w2)

    # plot as requested in assignment
    plot_training_and_test_data_graph(positive_train,negative_train,positive_test,negative_test,"Decision Boundary with test & train")

    # split correct / incorrect test predicitons
    Xcorrect, Xincorrect = split_by_correct_prediction(X_test[:,0],X_test[:,1],y_test, y_pred)

     # add decision boundary
    add_decision_boundary_to_plot(b,w1,w2)

    # plot correct / incorrect graph with decision boundary
    plot_correct_incorrect_graph(Xcorrect,Xincorrect,"Decision Boundary Correct/Incorrect")

def plot_splitted_data(positive,negative):
    plt.title('Visualisation of training data')
    plt.xlabel('first training feature')
    plt.ylabel('second training feature')
    plt.scatter(positive[:,0],positive[:,1], label="Positive Train",color='g',marker='+')
    plt.scatter(negative[:,0],negative[:,1], label="Negative Train",color='b')
    plt.legend()
    plt.show()



def add_decision_boundary_to_plot(b,w1,w2):
    """ 
    
    this code is derived from https://scipython.com/blog/plotting-the-decision-boundary-of-a-logistic-regression-model/?fbclid=IwAR0YH5_vGj3lcZe21YGX1EVrwuarUKH4FQAo72yMvfFjIywaJ6Y8lJUaNkI

    """

    # get intercept and gradient
    
    c,m = calculate_intercept_and_gradient_from_model_parameters(b,w1,w2)
    
    # Plot decision boundary.
    xmin, xmax = -1, 1
    ymin, ymax = -1, 1
    xd = np.array([xmin, xmax])
    yd = m*xd + c
    plt.plot(xd, yd, 'k', lw=1, ls='--', label="Decision Boundary")
    plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
    plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)

def calculate_intercept_and_gradient_from_model_parameters(b,w1,w2):
    """ 
    
    this code is derived from https://scipython.com/blog/plotting-the-decision-boundary-of-a-logistic-regression-model/?fbclid=IwAR0YH5_vGj3lcZe21YGX1EVrwuarUKH4FQAo72yMvfFjIywaJ6Y8lJUaNkI

    """

    # Calculate the intercept and gradient of the decision boundary.
    c = -b/w2
    m = -w1/w2
    return c, m

def plot_correct_incorrect_graph(Xcorrect,Xincorrect,title):
    plt.title(title)
    plt.xlabel('first training feature')
    plt.ylabel('second training feature')
    plt.scatter(Xcorrect[:,0],Xcorrect[:,1], label="Correct predictions",color='orange')
    plt.scatter(Xincorrect[:,0],Xincorrect[:,1], label="Incorrect Predictions",color='yellow')
    plt.legend()
    plt.show()

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

def main_b():
    """ Week 2 Machine Learning Assignment Q B

            Linear SVM classifiers

    """

#####################################################
#                                                   #
#                  Question B                       #
#                                                   #
#####################################################

    # adjust graph values so legibly
    plt.rc('font', size=14)
    plt.rcParams['figure.constrained_layout.use'] = True

    # Read in data    
    df = pd.read_csv('week2.csv')
    print(df.head())
    X1=df.iloc[: ,0]
    X2=df.iloc[: ,1]
    X=np.column_stack(( X1, X2 ))
    y=df.iloc[: , 2]

    # split data for test/train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    #split to visualise
    positive_train, negative_train = split(X_train[:,0],X_train[:,1],y_train)

    plot_splitted_data(positive_train,negative_train)

    # train SVM classifiers
    model_0001 = LinearSVC(C=0.001).fit(X_train, y_train)
    model_1 = LinearSVC(C=1).fit(X_train, y_train)
    model_1000 = LinearSVC(C=1000).fit(X_train, y_train)

    # all C = 0.001
    classes_0001 = model_0001.classes_
    coef_0001 = model_0001.coef_
    intercept_0001 = model_0001.intercept_

    print('classes C = 0.001: ,' ,classes_0001)
    print('coef  C = 0.001: ,' , coef_0001)
    print('intercept  C = 0.001: ,' ,intercept_0001)

    # all C = 1
    classes_1 = model_1.classes_
    coef_1 = model_1.coef_
    intercept_1 = model_1.intercept_

    print('classes C = 1: ,' ,classes_1)
    print('coef  C = 1: ,' , coef_1)
    print('intercept  C = 1: ,' ,intercept_1)

    # all C = 1000
    classes_1000 = model_1000.classes_
    coef_1000 = model_1000.coef_
    intercept_1000 = model_1000.intercept_

    print('classes C = 1000: ,' ,classes_1000)
    print('coef  C = 1000: ,' , coef_1000)
    print('intercept  C = 1000: ,' ,intercept_1000)


# FOR C = 0.0001

    y_pred_0001 = model_0001.predict(X_test)

     # Show the Confusion Matrix
    print(confusion_matrix(y_test, y_pred_0001))

     # get model paramete values and plot decision boundary  # from online article - https://scipython.com/blog/plotting-the-decision-boundary-of-a-logistic-regression-model/

    b = intercept_0001[0]
    w1, w2 = coef_0001.T

    # Plot decision boundary
    add_decision_boundary_to_plot(b,w1,w2)

    # compare predictions to actual to show results of model on test data
    positive_test,negative_test = split(X_test[:,0],X_test[:,1],y_pred_0001)

    plot_training_and_test_data_graph(positive_train,negative_train,positive_test,negative_test,'SVM Classifier Results C=0.001')

    Xcorrect, Xincorrect = split_by_correct_prediction(X_test[:,0],X_test[:,1],y_test, y_pred_0001)

    plot_correct_incorrect_graph(Xcorrect, Xincorrect, 'SVM Classifier Results C=0.001')


# FOR C = 1

    y_pred_1 = model_1.predict(X_test)

     # Show the Confusion Matrix
    print(confusion_matrix(y_test, y_pred_1))

     # get model paramete values and plot decision boundary  # from online article - https://scipython.com/blog/plotting-the-decision-boundary-of-a-logistic-regression-model/

    b = intercept_1[0]
    w1, w2 = coef_1.T

    # Plot decision boundary
    add_decision_boundary_to_plot(b,w1,w2)

    # compare predictions to actual to show results of model on test data
    positive_test,negative_test = split(X_test[:,0],X_test[:,1],y_pred_1)

    plot_training_and_test_data_graph(positive_train,negative_train,positive_test,negative_test,'SVM Classifier Results C=1')

    Xcorrect, Xincorrect = split_by_correct_prediction(X_test[:,0],X_test[:,1],y_test, y_pred_1)

    plot_correct_incorrect_graph(Xcorrect, Xincorrect, 'SVM Classifier Results C=1')

# FOR C = 1000

    y_pred_1000 = model_1000.predict(X_test)

     # Show the Confusion Matrix
    print(confusion_matrix(y_test, y_pred_1000))

     # get model paramete values and plot decision boundary  # from online article - https://scipython.com/blog/plotting-the-decision-boundary-of-a-logistic-regression-model/

    b = intercept_1000[0]
    w1, w2 = coef_1000.T

    # Plot decision boundary
    add_decision_boundary_to_plot(b,w1,w2)

    # compare predictions to actual to show results of model on test data
    positive_test,negative_test = split(X_test[:,0],X_test[:,1],y_pred_1000)

    plot_training_and_test_data_graph(positive_train,negative_train,positive_test,negative_test,'SVM Classifier Results C=1000')

    Xcorrect, Xincorrect = split_by_correct_prediction(X_test[:,0],X_test[:,1],y_test, y_pred_1000)

    plot_correct_incorrect_graph(Xcorrect, Xincorrect, 'SVM Classifier Results C=1000')

def main_c():
    """ Week 2 Machine Learning Assignment Q C

                Logistic Regression w/ additional features

    """

#####################################################
#                                                   #
#                  Question C                       #
#                                                   #
#####################################################

    # adjust graph values so legibly
    plt.rc('font', size=14)
    plt.rcParams['figure.constrained_layout.use'] = True

    # Read in data    
    df = pd.read_csv('week2.csv')
    print(df.head())
    X1=df.iloc[: ,0]
    X2=df.iloc[: ,1]
    y=df.iloc[: , 2]
    print(y)

    # add additional features by squaring
    X1sq = X1 * X1
    X2sq = X2 * X2
    X=np.column_stack(( X1, X2, X1sq, X2sq ))

    # split data for test/train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = LogisticRegression()
    model.fit(X, y)
    y_pred = model.predict(X_test)

# all atributes
    classes = model.classes_
    coef = model.coef_
    intercept = model.intercept_

    print('classes : ,' ,classes)
    print('coef : ,' ,coef)
    print('intercept : ,' ,intercept)

    # Show the Confusion Matrix
    print(confusion_matrix(y_test, y_pred))

    #split to visualise
    positive_train, negative_train = split(X_train[:,0],X_train[:,1],y_train)

    # compare predictions to actual to show results of model on test data
    positive_test,negative_test = split(X_test[:,0],X_test[:,1],y_pred)

    plot_training_and_test_data_graph(positive_train,negative_train,positive_test,negative_test,'Logistic regression with squared features')

    Xcorrect, Xincorrect = split_by_correct_prediction(X_test[:,0],X_test[:,1],y_test, y_pred)

    plot_correct_incorrect_graph(Xcorrect,Xincorrect,'Logistic regression with squared features')

    # Compare against baseline that always predicts most common case
    most_common_case = get_most_common_case(y)

    print('most common case : ', most_common_case)

    # fill with most common case
    baseline_prediction = np.ones(len(y_test))

    # compare predictions to actual to show results of model on test data
    positive_test,negative_test = split(X_test[:,0],X_test[:,1],baseline_prediction)

    plot_training_and_test_data_graph(positive_train,negative_train,positive_test,negative_test,'Baseline Comparison')

    Xcorrect, Xincorrect = split_by_correct_prediction(X_test[:,0],X_test[:,1],y_test, baseline_prediction)

    plot_correct_incorrect_graph(Xcorrect,Xincorrect,'Baseline Comparison')

    # Show the Confusion Matrix
    print(confusion_matrix(y_test, baseline_prediction))


    # Try to plot decision boundary

def get_most_common_case(y):
    sum = np.sum(y) 
    if sum > 0:
        return 1
    if sum < 0:
        return -1
    return 0




if __name__ == "__main__":
    main_a()
    main_b()
    main_c()