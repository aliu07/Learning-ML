import math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients

# Load our data set
x_train = np.array([1.0, 2.0]) # features
y_train = np.array([300.0, 500.0]) # target value

# Function to calculate the cost (error)
def compute_cost(xs, ys, w, b):
    m = xs.shape[0] # Get length of array of xs
    cost = 0

    for i in range(m):
        f_wb = w * xs[i] + b
        cost += (f_wb - ys[i])**2
    
    total_cost = 1 / (2 * m) * cost
    return total_cost



def compute_gradient(xs, ys, w, b):
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    m = xs.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * xs[i] + b
        dj_dw += (f_wb - ys[i]) * xs[i]
        dj_db += (f_wb - ys[i])

    dj_dw *= 1 / m # Denote partial derivative w.r.t. w - d/dw J(w, b) - as dj_dw
    dj_db *= 1/ m

    return dj_dw, dj_db



def gradient_descent(xs, ys, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient
      
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] 
      """

    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient
        dj_dw, dj_db = compute_gradient(xs, ys, w, b)

        # Update parameters according to linear regression with 2 parameters
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000: # Prevent resource exhaustion
            J_history.append(cost_function(xs, ys, w, b))
            p_history.append(gradient_function(xs, ys, w, b))
        # Print cost at intervals of 1/10ths or every iteration if < 10
        if i % math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e} ",
                  f"w: {w: 0.3e}, b: {b: 0.5e}")
    
    return w, b, J_history, p_history # return values for graphing



if __name__ == '__main__':
    # Initialize parameters
    w_init = 0
    b_init = 0
    # Gradient descent settings
    iterations = 10000
    tmp_alpha = 1.0e-2
    # Run gradient descent
    w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)

    print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

    # Plot cost versus iteration  
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
    ax1.plot(J_hist[:100])
    ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
    ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
    ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
    ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step') 
    plt.show()

    # Making predictions using optimized model
    print(f"1000 sqft house prediction {w_final*1.0 + b_final:0.1f} Thousand dollars")
    print(f"1200 sqft house prediction {w_final*1.2 + b_final:0.1f} Thousand dollars")
    print(f"2000 sqft house prediction {w_final*2.0 + b_final:0.1f} Thousand dollars")  

    # Plotting predictions
    fig, ax = plt.subplots(1,1, figsize=(12, 6))
    plt_contour_wgrad(x_train, y_train, p_hist, ax)
    
    fig, ax = plt.subplots(1,1, figsize=(12, 4))
    plt_contour_wgrad(x_train, y_train, p_hist, ax, w_range=[180, 220, 0.5], b_range=[80, 120, 0.5],
    contours=[1,5,10,20],resolution=0.5)

    plt.show()