import numpy as np
import matplotlib.pyplot as plt
import copy, math

def sigmoid(X):
    """
    Computes the sigmoid value for each element of a vector

    Args:
      X (ndarray(m,n)): Data, m examples with n features
    Returns
      X' (ndarray(m,n)): The transformed vector
    """
    return 1 / (1 + np.exp(-X))



def compute_gradient_logistic(X, y, w, b): 
    """
    Computes the gradient for logistic regression 
 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
    Returns
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b. 
    """
    m, n = X.shape
    dj_dw = np.zeros(n,)
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i, j]
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_db, dj_dw



def compute_cost_logistic(X, y, w, b):
    """
    Computes the cost

    Args: 
      X (ndarray (m,n)   : Data, m examples with n features
      y (ndarray (m,))   : target values
      w (scalar)         : Value of weight
      b (scalar)         : Value of bias
    Returns:
      cost (float)       : Cost given model parameters
    """
    m = X.shape[0]
    cost = 0. 

    for i in range(m):
        f_wb = sigmoid(np.dot(X[i], w) + b)
        cost = cost + y[i] * np.log(f_wb) + (1-y[i]) * np.log(1 - f_wb)

    cost = -1 * cost / m
    return cost




def gradient_descent(X, y, w_in, b_in, alpha, num_iters): 
    """
    Performs batch gradient descent
    
    Args:
      X (ndarray (m,n)   : Data, m examples with n features
      y (ndarray (m,))   : target values
      w_in (ndarray (n,)): Initial values of model parameters  
      b_in (scalar)      : Initial values of model parameter
      alpha (float)      : Learning rate
      num_iters (scalar) : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,))   : Updated values of parameters
      b (scalar)         : Updated value of parameter 
    """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in

    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)   

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db 

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( compute_cost_logistic(X, y, w, b) )

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
        
    return w, b, J_history         #return final w,b and J history for graphing


if __name__ == '__main__':
    # Training data sets
    X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    w_tmp  = np.zeros_like(X_train[0])
    b_tmp  = 0.
    alph = 0.1
    iters = 10000

    w_out, b_out, _ = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters) 
    print(f"\nupdated parameters: w:{w_out}, b:{b_out}")

    fig, ax = plt.subplots()
    x_vals = X_train[:, 0]
    y_vals = X_train[:, 1]
    class_0 = X_train[y_train == 0]
    class_1 = X_train[y_train == 1]
    plt.scatter(class_0[:, 0], class_0[:, 1], marker='o', label='y=0')
    plt.scatter(class_1[:, 0], class_1[:, 1], marker='x', label='y=1')

    # Plot decision boundary
    x_boundary = np.linspace(0, 3.5, 100)
    y_boundary = -(w_out[0] * x_boundary + b_out) / w_out[1]
    plt.plot(x_boundary, y_boundary, color='red', label='Decision Boundary')

    ax.set_xlabel('$x_0$', fontsize=12)
    ax.set_ylabel('$x_1$', fontsize=12)
    plt.legend()
    plt.show()