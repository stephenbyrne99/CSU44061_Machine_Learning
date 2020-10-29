# id:15-5170.5--135 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
import math
# import time

def normalise_array(input_arr):

    """ Returns a normalised array. 

    This function takes a numpy array, and normalises each element.
    """

    max_value = np.max(input_arr)
    min_value = np.min(input_arr)
    normalised_array = np.array([])
    for elem in input_arr:
        normalised_elem = (elem - min_value) / (max_value - min_value)
        normalised_array = np.append(normalised_array,normalised_elem)
    return normalised_array


def plot_line(theta0, theta1, X, y, title='Regression model trained with gradient descent'):

    """ Plots a line where y = theta1x + theta0 . 

    """

   # get max X values to stay within bounds 
    max_x = np.max(X) 
    min_x = np.min(X) 

    # get evenly spaced x points to map to y outputs - could use any number as drawing straight line or m
    xplot = np.linspace(min_x, max_x, 10) 

    # calculate hypothesis 
    yplot = hypothesis(theta0,theta1,xplot)

    plt.plot(xplot, yplot, color='r', label='Regression Line')
    plt.legend()
    plt.title(title)
    plt.xlabel('Normalised X')
    plt.ylabel('Normalised y')
    plt.scatter(X,y, label="Data",color='b')
    plt.show()

 

def hypothesis(theta0, theta1, x):
    """ Calculate hypothesis. 

    Maps inputs to estimate outputs. 
    """

    return theta0 + (theta1*x) 


def get_cost(theta0, theta1, X, y):

    """ Finds cost value using Mean Squared Error . 

        Finding the cost of is the difference between estimated values, or the hypothesis, and the real values

        Uses X input to calculate the hypothesis for each value of xi
        using given parameter values.
        Cost then found by MSE by taking away from actual value.
    """

    cost = 0 
    # iterate over both 
    for (xi, yi) in zip(X, y):
                            #estimated                #real
        cost += 0.5 * ((hypothesis(theta0, theta1, xi) - yi)**2)
    return cost


def derivatives(theta0, theta1, X, y):
    """ Handles bulk of algo work, cost minimisation / partial derivatives

    """

    dtheta0 = 0
    dtheta1 = 0
    for (xi, yi) in zip(X, y):
        dtheta0 += hypothesis(theta0, theta1, xi) - yi
        dtheta1 += (hypothesis(theta0, theta1, xi) - yi)*xi

    dtheta0 /= len(X) # divide by m
    dtheta1 /= len(X) # divide by m

    return dtheta0, dtheta1

def minimise(theta0, theta1, X, y, alpha):
    """ updates parameters. 
    
    """

    dtheta0, dtheta1 = derivatives(theta0, theta1, X, y)
    theta0 = theta0 - (alpha * dtheta0)
    theta1 = theta1 - (alpha * dtheta1)

    return theta0, theta1
    

def linear_regression(X, y, epochs, a):
    """ Trains linear regression model using gradient descent, 
        returns theta0 and theta1 (parameter values) upon completion.

    """

    theta0 = np.random.rand()
    theta1 = np.random.rand()
 
    for i in range(0, epochs):

        # print graph every 100 iterations showing progress
        if i % 100 == 0:
            plot_line(theta0, theta1, X, y)

            # print(get_cost(theta0, theta1, X, y))
 
        theta0, theta1 = minimise(theta0, theta1, X, y, a)

    return theta0, theta1


def linear_regression_b(X, y, epochs, a):
    """ Trains linear regression model using gradient descent, 
        returns cost array on completion (value of cost for each iteration).

        
    """

    theta0 = np.random.rand()
    theta1 = np.random.rand()
    cost_arr = []
 
    for i in range(0, epochs):
 
        theta0, theta1 = minimise(theta0, theta1, X, y, a)
        cost_arr.append(get_cost(theta0, theta1, X, y))

    return cost_arr





def main():

    """ Week 1 Machine Learning Assignment 


    """

#####################################################
#                                                   #
#                  Question A                       #
#                                                   #
#####################################################

    # adjust graph values so legibly
    plt.rc('font', size=14)

    # Read in data (i)
    df = pd.read_csv("week1.csv",comment="#")
    print(df.head())
    X=np.array(df.iloc[:,0])
    y=np.array(df.iloc[:,1])

    plt.scatter(X,y)
    plt.title('Week 1 Data')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()
   

    # Normalise it (ii)
    normalised_y = normalise_array(y)
    print(normalised_y)
    normalised_X = normalise_array(X)
    print(normalised_X)

    plt.scatter(normalised_X,normalised_y)
    plt.title('Normalised Week 1 Data')
    plt.xlabel('Normalised X')
    plt.ylabel('Normalised y')
    plt.show()

    # Use gradient descent to train a linear model (iii)
    theta0, theta1 = linear_regression(normalised_X,normalised_y,1000,0.1)

    #####################################################
    #                                                   #
    #                  Question B                       #
    #                                                   #
    #####################################################
    
    #(i) Try a range of learning rates α in your gradient descent algorithm, e.g. 0.001, 0.01, 0.1, 
        # and plot how the cost function J(θ) changes over time. Hint: for 
        # v small learning rates the cost function should decrease v slowly, for v large 
        # learning rates the cost function may not converge.

    # get max X values to stay within bounds 
    cost_arr = linear_regression_b(normalised_X,normalised_y,1000,0.1)

    # get evenly spaced x points to map to y outputs - must be same as epochs
    xplot = np.linspace(0, 1000, 1000) 

    plt.plot(xplot, cost_arr, color='r', label='Cost')
    plt.legend()
    plt.title('How J0 changes over time')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()
    

    #(ii) Report the parameter values of the linear regression model after it has been trained on your downloaded data.
    print('theta0: ', theta0)
    print('theta1: ', theta1)

    #(iii) Also report the value of the cost function for the trained model.

    print('Value of final cost: ' ,get_cost(theta0,theta1,normalised_X,normalised_y))

        # Compare with 
        # the value of the cost function for a baseline model that always predicts a constant
        # value (pick a reasonable value based on inspection of your data).

    plot_line(0.5,0,normalised_X,normalised_y,'Baseline model') # changed the plot function for this part to change labels in submitted report.

    print('Value of final cost: ' ,get_cost(0.5,0.5,normalised_X,normalised_y))
    
    
    #(iv) Now use sklearn to train a linear regression model on your data. 

    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    LR = LinearRegression()
    LR.fit(normalised_X.reshape(-1,1),normalised_y)
    prediction = LR.predict(normalised_X.reshape(-1,1))

    plt.plot(normalised_X,prediction,label="Linear Regression",color='r')
    plt.scatter(normalised_X,normalised_y,label="Data",color='b')
    plt.title('Regression model trained sklearn')
    plt.legend()
    plt.show()
    


if __name__ == "__main__":
    main()