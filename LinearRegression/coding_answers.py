import numpy as np
import matplotlib.pyplot as plt

####################################Creating Data#############################################
N = 25
X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y = np.cos(10*X**2) + 0.1 * np.sin(100*X)


############################Functions to be used in questions#################################


def poly_create_Phi(X, k):
    """Creates the Phi matrix for datapoints X, for a polynomial basis function of order k.

    :param X: (25 x 1) numpy array containing the x-data for the 25 points.
           k: Integer giving the order of the polynomial basis function for which Phi is constructed.

    :return Phi: (25 x k+1) numpy array where each row contains a particular x-datum raised to all powers from 0 to k."""
    
    Phi = np.zeros((np.size(X), k+1))
    for phi_col in range(k+1):
        Phi[:,phi_col] = (X ** (phi_col)).flatten()
        
    return Phi


def trig_create_Phi(X, k):
    """Creates the Phi matrix for datapoints X, for a trigonometric basis function of order k.

    :param X: (25 x 1) numpy array containing the x-data for the 25 points
           k: Integer giving the order of the polynomial basis function for which Phi is constructed

    :return Phi: (25 x 2k+1) numpy array where each row n corresponds to a particular x-datum x_n, with columns [1, sin(2π*1*x_n), cos(2π*1*x_n), sin(2π*2*x_n), ..., cos(2π*k*x_n)]."""

    Phi = np.ones((np.size(X), 2 * k + 1))
    for phi_col in range(k):
        Phi[:, 2 * phi_col + 1] = np.sin(2*np.pi*X*(phi_col+1)).flatten()
        Phi[:, 2 * phi_col + 2] = np.cos(2*np.pi*X*(phi_col+1)).flatten()
        
    return Phi


def get_w_ml(Phi, Y):
    """Calculates the parameters w_ml which, when multiplied by the basis functions, are most likely to describe the function from which datapoints stem
    (assuming y-data are normally distributed around the mean given by the aforementioned function of x-data).

    :param Phi: Numpy array giving the basis functions for each x-datum.
           Y: (25 x 1) numpy array containing y-data.

    :return w_ml: Numpy column vector array giving the parameters which, when multiplied by the basis functions, are most likely to describe the relation from which datapoints stem."""
    
    w_ml = (np.linalg.inv(Phi.T @ Phi)) @ (Phi.T @ Y)
    
    return w_ml


def get_sigma_ml(Phi, w_ml, Y):
    """Calculates the common variance with which y-data are distributed around functions of their corresponding x-data, where these functions are given by combinations of basis functions
    of corresponding x-data multiplied by parameters w_ml.

    :param Phi: Numpy array giving the basis functions for each x-datum.
           w_ml: Numpy column vector array giving the parameters which, when multiplied by the basis functions, are most likely to describe the relation from which datapoints stem.
           Y: (25 x 1) numpy array containing y-data.

    :return sigma_ml: (1 x 1) numpy array giving the variance with which y-data are normally distributed around the assumed functions of corresponding x-data."""

    N = np.size(Y)
    err = Y - (Phi @ w_ml)
    sigma_ml = (err.T @ err) / N
    
    return sigma_ml


def est_mean(basis, x_range, X, Y, k):
    """Estimates the mean around which y-data are normally distributed by assuming a weighted, (by optimised parameters w_ml), sum of either polynomial or trig basis functions
    of corresponding x-data of order k. Trains the parameters w_ml using data X, Y, and uses them to estimate the mean of y-data corresponding to x-data in x_range.

    :param basis: String specifying whether polynomial or trigonometric basis functions are to be considered.
           x_range: Numpy column vector array giving the x-data at which the means of hypothetical y-data are to be estimated.
           X: (25 x 1) numpy array containing x-data for the 25 points to be used for training parameters.
           Y: (25 x 1) numpy array containing y-data for the 25 points to be used for training parameters.
           k: Integer giving the order of the basis functions to be considered in training.

    :return: y_range_mean: Numpy vector array containing the mean values around which ys corresponding to xs in x_range are expected to be normally distributed"""

    if basis == "polynomial":
        Phi = poly_create_Phi(X, k)
        Phi_x_range = poly_create_Phi(x_range, k)
        
    elif basis == "trigonometric":
        Phi = trig_create_Phi(X, k)
        Phi_x_range = trig_create_Phi(x_range, k)

    w_ml = get_w_ml(Phi, Y)
    sigma_ml = get_sigma_ml(Phi, w_ml, Y)

    y_range_mean = Phi_x_range @ w_ml
    
    return y_range_mean


def plot_est_means(basis, domain, X, Y, k_all):
    """Plots the estimated means around which ys are expected to be normally distributed for their corresponding xs in domain.

    :param basis: String specifying whether polynomial or trigonometric basis functions are to be considered.
           domain: list specifying the domain of xs, (x_range), of xs for which means for corresponding ys are to be estimated.
           X: (25 x 1) numpy array containing x-data for the 25 points to be used for training parameters.
           Y: (25 x 1) numpy array containing y-data for the 25 points to be used for training parameters.
           k_all: list giving all the orders, k, of the basis functions to be considered in training for producing estimated means for ys."""

    x_range = np.linspace(domain[0], domain[1], 200)
    
    plt.figure()
    plt.plot(X, Y, '+', label = 'data')

    for k in k_all:
        y_range_mean = est_mean(basis, x_range, X, Y, k)
        plt.plot(x_range, y_range_mean, label = 'k = {}'.format(k))

    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.ylim([-3,3])
    plt.xlim([x_range[0], x_range[-1]])
    plt.legend()
    
    return 0


def split_data_one_out(X, Y, idx_out):
    """Separates one of the 25 data pairs to be used as test datum in cross validation. The remaining 24 are to be used as training data.
    :param X: (25 x 1) numpy array containing x-data for the 25 points.
           Y: (25 x 1) numpy array containing y-data for the 25 points.
           idx_out: index of the datum to be separated and used as test in cross validation.

    :return x_test: (1 x 1) numpy array containing x-value of datum to be used as test in cross validation.
            y_test: (1 x 1) numpy array containing y-value of datum to be used as test in cross validation.
            X_train: (24 x 1) numpy array containing x-values of data to be used as training in cross validation.
            Y_train: (24 x 1) numpy array containing y-values of data to be used as training in cross validation."""
    
    x_test = X[idx_out - 1]
    y_test = Y[idx_out - 1]

    X_train = np.delete(X, idx_out - 1, 0)
    Y_train = np.delete(Y, idx_out - 1, 0)

    return x_test, y_test, X_train, Y_train


def get_trig_MSE_one_fold_one_order(X, Y, k, idx_out):
    """Finds the mean squared error, (MSE), of the predicted value of y corresponding x_test from actual y_test. Here y is predicted by the mean value corresponding to x_test,
    estimated using a combination of trigonometric functions of order k, weighted by parameters w_opt, trained by X_train, Y_train.

    :param X: (25 x 1) numpy array containing x-data for the 25 points.
           Y: (25 x 1) numpy array containing y-data for the 25 points.
           idx_out: index of the datum to be separated and used as test in cross validation.

    :return sigma_opt: (1 x 1) numpy array containing the expected variance of ys from their estimated means.
            MSE: (1 x 1) numpy array containing the mean squared error of the predicted y for x_test from actual y_test."""
    
    x_test, y_test, X_train, Y_train = split_data_one_out(X, Y, idx_out)
    Phi_train = trig_create_Phi(X_train, k)
    w_opt = get_w_ml(Phi_train, Y_train)
    sigma_opt = get_sigma_ml(Phi_train, w_opt, Y_train)

    Phi_test = trig_create_Phi(x_test, k)
    y_pred = Phi_test @ w_opt
    MSE = ((y_test - y_pred) ** 2)

    return sigma_opt, MSE


def cross_validate_trig_one_order(X, Y, k):
    """Finds the average variance of the models trained by all 25 combinations of 24 data pairs, and the average of all 25 MSEs of predicted ys by each model from the actual y_test.
    All this is done for models trained with trigonometric basis functions of one particular order k.

    :param X: (25 x 1) numpy array containing x-data for the 25 points.
           Y: (25 x 1) numpy array containing y-data for the 25 points.
           k: Integer giving the order of the basis functions of the models to be trained.

    :return sigma_opt_ave[0,0]: Integer giving the average variance of all models.
            MSE_ave[0,0]: Integer giving the average MSE between all y-predicted and y_test pairs considered."""

    sigma_opt_all = []
    MSE_all = []

    for idx_out_minus_one in range(len(X.flatten())):
        idx_out = idx_out_minus_one + 1
        sigma_opt, MSE = get_trig_MSE_one_fold_one_order(X, Y, k, idx_out)
        sigma_opt_all.append(sigma_opt)
        MSE_all.append(MSE)

    sigma_opt_ave = sum(sigma_opt_all) / len(sigma_opt_all)
    MSE_ave = sum(MSE_all) / len(MSE_all)

    return sigma_opt_ave[0,0], MSE_ave[0,0]



###########################Question 1#####################################
domain = [-0.3, 1.3]
basis = "polynomial"
k_all = [0, 1, 2, 3, 11]

w_opt = []
sigma_opt = []
for k in k_all:
    Phi = poly_create_Phi(X, k)
    w_opt_k = get_w_ml(Phi, Y)
    sigma_opt_k = get_sigma_ml(Phi, w_opt_k, Y)
    w_opt.append(w_opt_k)
    sigma_opt.append(sigma_opt_k)

print(w_opt)
print(sigma_opt)

plot_est_means(basis, domain, X, Y, k_all)
plt.show()

###########################Question 2#########################################
domain = [-1, 1.2]
basis = "trigonometric"
k_all = [1, 11]

w_opt = []
sigma_opt = []
for k in k_all:
    Phi = poly_create_Phi(X, k)
    w_opt_k = get_w_ml(Phi, Y)
    sigma_opt_k = get_sigma_ml(Phi, w_opt_k, Y)
    w_opt.append(w_opt_k)
    sigma_opt.append(sigma_opt_k)

print(w_opt)
print(sigma_opt)

plot_est_means(basis, domain, X, Y, k_all)
plt.show()

###########################Question 3#####################################
k_all = [0,1,2,3,4,5,6,7,8,9,10]

sigma_opt_ave_all = []
MSE_ave_all = []

for k in k_all:
    sigma_opt_ave_k, MSE_ave_k = cross_validate_trig_one_order(X, Y, k)
    sigma_opt_ave_all.append(sigma_opt_ave_k)
    MSE_ave_all.append(MSE_ave_k)

plt.plot(k_all, MSE_ave_all, 'x', label = 'MSE')
plt.plot(k_all, sigma_opt_ave_all, 'o', label = 'sigma^2')
plt.xlabel("Trigonometric basis order $k$")
plt.ylabel("Variance / MSE")
plt.grid()
plt.legend()
plt.show()









    

