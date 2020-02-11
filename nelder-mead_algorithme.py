# Do not delete the following import lines
import numpy as np

def centeroidnp(arr):
    length = arr.shape[0]
    return (1/length)*np.sum(arr,axis=0)

def order_X(arr,f):
    l = np.array([f(elt) for elt in arr])
    index_sorted = np.argsort(l)
    arr = arr[index_sorted]
    return arr

def nelderMeadStudents(costFunction, initialSimplex, maxIter, saveData =
                       False, deltaXConv = 0.01, alpha = 1., gamma= 2.,
                       rho = 0.5, sigma = 0.5):
    """
    Nelder Mead algorithm implementation:
    The Nelder-Mead method or downhill simplex method or amoeba method
    is a commonly applied numerical method used to find the minimum or
    maximum of an objective function in a multidimensional space. It 
    is applied to nonlinear optimization problems for which 
    derivatives may not be known. However, the Nelder-Mead technique 
    is a heuristic search method that can converge to non-stationary 
    points on problems that can be solved by alternative methods.
    
    The implementation is given by the algorithm describen in:
    https://en.wikipedia.org/wiki/Nelder-Mead_method

    input parameters:
    - costFunction
    Callable cost fuction that can be called on a point just by doing:
    costFunction(x) where x is a point (ndimensions)
    - initialSimplex: 
    np array of size (ndimensions+1, ndimensions) containing the 
    points that form the initial simplex
    - maxIter:
    Maximum number of iterations allowed
    - saveData:
    (optional: default value False) boolean defining whether or not 
    the data from all the iteraions should be stored and returned
    - deltaXConv:
    (optional: default 0.01) iteration step stopping criterion
    - alpha:
    (optional: default 1.) reflection parameter > 0
    - gamma:
    (optional: default 2.) expansion factor > 1
    - rho:
    (optional: default 0.5) contraction parameter 0 < rho <= 0.5
    - sigma:
    (optional: default 0.5) shrink parameter 
    
    output:
    - best_point
    best point at the end
    - data (optional)
    if saveData parameter is True, the there is a second output. It is a 
    dictionnary containing:
    data["niteraions"] iteration number (np array)
    data["x_best"] list of best solution at each iteration
    data["delta_x_best"] list of current dx
    data["f_x_best"] list of the 
    """

    #Order according to the values at the vertices
    X = initialSimplex
    
    if saveData :
        data = {"niteraions":[],"x_best":[],"delta_x_best":[],"f_x_best":[]}
    iter = 1


    
    while iter <= maxIter :
        X = order_X(X,costFunction)
    #Calculate the centroid of all points except xn+1
        x0 = centeroidnp(X[:-1])

    #Reflection        
        xr = x0 + alpha * (x0-X[-1])
        if costFunction(X[0]) <= costFunction(xr)  and costFunction(xr) <= costFunction(X[-2]) :
            dx = X[-1] - xr
            X[-1] = xr

    #Expansion
        elif costFunction(xr) <= costFunction(X[0]) :
            xe = x0 + gamma * (xr - x0)
            if costFunction(xe) <= costFunction(xr) :
                dx = X[-1] - xe
                X[-1]=xe
                

            else :
                dx = X[-1] - xr
                X[-1] = xr


    #Contraction
        else :
            xc = x0 + rho * (X[-1] - x0)
            if costFunction(xc) <= costFunction(X[-1]) :
                dx = abs(X[-1]-xc)
                X[-1] = xc

    #Shrink
            else :
                for i in range(1,len(X)): 
                    X[i] = X[0] + sigma * (X[i] - X[0])

        if saveData :
            data["niteraions"].append(iter)
            
            data["delta_x_best"].append(dx)
            
            cost = np.array([costFunction(elt) for elt in X])
            index_best_point = np.argsort(cost)[0]
            
            data["x_best"].append(X[index_best_point])
            data["f_x_best"].append(costFunction(X[index_best_point]))


            
        iter+= 1
        if np.std(cost) < deltaXConv :
            break
        
        
    #print(X,'\n',cost) 
    return X[index_best_point] , data
