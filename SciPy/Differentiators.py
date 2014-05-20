
import numpy as np

def twoPtForwardDiff(x,y):
    """This function takes the first element to the last element in the array and differentiates them. 
    The last element is differentiated by taking the difference between the last two y's over the last two x's"""
    dydx = np.zeros(y.shape,float)
    dydx[0:-1] = np.diff(y)/np.diff(x)
    dydx[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])
    return dydx

def twoPtCenteredDiff(x,y):
    """This function takes the first element in the array and differentiates it by taking the first two y's over the first two x's.
    The second element to the last element is differentiated by subtrating all the elements on the y-axis between the third element
    and the second to last element over the difference of the elements on the x-axis between the third element and the second to
    last element. The last element is differentiated by taking the difference between the last two y's over the last two x's."""
    dydx = np.zeros(y.shape,float)
    dydx[1:-1] = (y[2:] - y[:-2])/(x[2:] - x[:-2])
    dydx[0] = (y[1]-y[0])/(x[1]-x[0])
    dydx[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])
    return dydx

def fourPtCenteredDiff(x,y):
    """This is a higher order differential equation from the Taylor series expansion and it is used to define the derivative to its
    finest points. It takes the third element to the second to last element to refine the derivative along those elements."""
    dydx = np.zeros(y.shape,float)
    dydx[2:-2] = (y[0:-4] - 8*y[1:-3] + 8*y[3:-1] - y[4:])/(12*.1)
    dydx[0] = (y[1]-y[0])/(x[1]-x[0])
    dydx[1] = (y[2]-y[1])/(x[2]-x[1])
    dydx[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])
    dydx[-2] = (y[-2] -y[-3]) / (x[-2] - x[-3])
    return dydx