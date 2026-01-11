#import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats as stats

#---------------------------------------------------------------------------------------------------
#Integral Normal Distribution 
def int_norm(x):
    mean = 0    # Mean (loc)
    std = 1     
    y = stats.norm.cdf(x, loc=mean, scale=std)
    return y

#---------------------------------------------------------------------------------------------------
# Where I is called mechanical error or degree of imperfection
# I is usually in the range of 0.15-0.25
# d is the density
# d50 is the density of separation
# p - partition, percentage of coal particles reported to the sink product
# intial guess of parameters: p = [0.2, 1.5]
def model_int_normal_forJig(d, *p):
    I = p[0]
    d50 = p[1]
    
    t = 0.6745 * np.log((d - 1)/(d50 - 1))
    t = 2 * t / np.log((1 + I)/(1 - I))
    p = 100 * int_norm(t)
    return p

#---------------------------------------------------------------------------------------------------
# Logistics model
# where x0 is the separation density. k is parameter to scale the axis
# initial guess of the parameters: p=[10, 1.5] for a partition curve
def model_Logistics(x, *p):	
    k = p[0]
    x0 = p[1]

    f = 100 *  1 /( 1 +  np.exp( -k * (x-x0) ) )  
    return f

#---------------------------------------------------------------------------------------------------
# Modified Logistics model
# Initial guess of the parameters: p=[15, 1.5, -1, 0.1] for partition curves fitting
def model_Modified_Logistics(x, *p):	
    k = p[0]
    x0 = p[1]
    a = p[2]
    b = p[3]

    f = 100 * ( a + b * x + 1 / (1 +  np.exp(-k * (x-x0))) )     
    return f

#---------------------------------------------------------------------------------------------------
# Modified Hyperbolic Tangent model
# Initial guess of the parameters: p = [10, 1.5, 1, -1, -0.1] for partition curves 
def	model_Modified_HyperbolicTangent(x, *p):
		# y = 100 (a + b * tanh( k * ( x-x0 ) ) + c * x )              //this is the one used in the Coal Preparation Software Package
		# y = 100 (1.0 - (a + b * tanh( k * ( x-x0 ) ) + c * x ) )     //I modified it to make y increase with x as a density or ash

		k = p[0]
		x0 = p[1]
		a = p[2]
		b = p[3]
		c = p[4]

		t = k * (x - x0)
		e = np.exp(t)
		# f = 100 * (a + b * (e * e - 1) / (e * e + 1) + c * x);       //the one used by the coal preparation software package 
		f = 100 * (1.0 - (a + b * (e * e - 1) / (e * e + 1) + c * x))
		return f

#---------------------------------------------------------------------------------------------------
# curve fit to the mode and plot the graph
# this is the initial edition. Please curve_fit_modelw
def curve_fit_model(model, xdata, ydata, p, title):
    start_time = time.perf_counter()
    popt, pcov, infodict, errmsg, ier = curve_fit(model, xdata, ydata, p0=p, full_output=True)
    # Get the number of function evaluations
    iteration_count = infodict['nfev']
    print(f"Number of function calls (iterations): {iteration_count}")
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"The code block executed in {elapsed_time:.4f} seconds")    

    print (popt)
    print (pcov)

    x0 = np.linspace(1.2, 2.4, 120)
    y0 = model(x0, *popt)

    x = []
    y = []

    for i in range(len(x0)):
        if y0[i] >= 0 and y0[i] <= 100 :
            x.append(x0[i])
            y.append(y0[i])        

    plt.scatter(xdata,ydata,label='Data')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.plot(x, y, 'r-', label='Fit')
    plt.legend()
    plt.xlabel("Coal Density")
    plt.ylabel("Partition (%)")
    plt.title("Model: " + title)
    xtick_positions = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4]
    plt.xticks(xtick_positions)  
    ytick_positions = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    plt.yticks(ytick_positions)
    manager = plt.get_current_fig_manager()
    manager.window.title("Coal Partition Curve")

    plt.show()
        
#---------------------------------------------------------------------------------------------------
# curve_fit to the model with sigma as a weight assigned and draw the graph 
def curve_fit_modelw(model, xdata, ydata, sigma, p, title): 
    if len(sigma) == 0:
        sigma = [1] * len(xdata) 
    #popt, pcov = curve_fit(model, xdata, ydata, sigma=sigma, p0=p, xtol=1e-6)
    popt, pcov = curve_fit(model, xdata, ydata, sigma=sigma, p0=p)    
    print (popt)
    #print (pcov)
        
    x0 = np.linspace(1.2, 2.4, 120)
    y0 = model(x0, *popt)

    x = []
    y = []

    for i in range(len(x0)):
        if y0[i] >= 0 and y0[i] <= 100 :
            x.append(x0[i])
            y.append(y0[i])        

    fig, ax = plt.subplots()

    plt.scatter(xdata,ydata,label='Data')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.plot(x, y, 'r-', label='Fit')
    plt.legend()
    plt.xlabel("Coal Density")
    plt.ylabel("Partition (%)")
    plt.title("Model: " + title)
    xtick_positions = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4]
    plt.xticks(xtick_positions)  
    ytick_positions = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    plt.yticks(ytick_positions)    
    
    manager = plt.get_current_fig_manager()
    manager.window.title("Coal Partition Curve")

    plt.show()
    
"""
def calculateSquareRootOfSumOfSquareResiduals(model, p):
    y_predicted = model(xdata, p)
    r = (ydata - y_predicted)
    dia_sigma = torch.diag(torch.tensor(sigma))
    inv_sigma = torch.linalg.inv(dia_sigma)
    print(inv_sigma)
    print(torch.tensor(r))
    n = len(xdata)
    rr = torch.tensor(r) @ inv_sigma.double()
    ss_r = np.sum(rr.numpy()**2)
    rmse = np.sqrt(ss_r/n)
    print("The square root of the sume of the squares of residuals:", rmse) 
"""

#---------------------------------------------------------------------------------------------------
# Example model fitting: coal separation/partition curves    
xdata = np.array([1.25, 1.35, 1.45, 1.55, 1.7, 2.0])            # densities
ydata = np.array([2.0, 10.0, 48.0, 75.0, 92.0, 100.0])          # percentage of coal reporting to the sink product (partition)

xdata = np.array([1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.90, 2.20])              # densities
ydata = np.array([0.16, 1.90, 10.84, 34.59, 64.25, 83.68, 95.12, 99.65])        # percentage of coal reporting to the sink product

print(xdata)
print(ydata)

# same data set can be fitting to different models
model = model_int_normal_forJig
p = [0.2, 1.5]

model = model_Logistics
p=[10, 1.5]

model = model_Modified_Logistics
p=[15, 1.5, -1, 0.1]

model = model_Modified_HyperbolicTangent
p = [10, 1.5, 1, -1, -0.1]

#title = model.__name__
#curve_fit_model(model_Logistics, xdata, ydata, p=p, title=title)

#sigma = [0.1, 0.5, 1., 1., 1., 1., 0.5, 0.2 ]         # for coal float and sink test, at the two ends of density intervals, it should be more accurate as more percentage of coal fall into the two ends of densities
sigma = [1.0] * len(xdata)
title = model.__name__
curve_fit_modelw(model, xdata, ydata, sigma, p=p, title=title)

