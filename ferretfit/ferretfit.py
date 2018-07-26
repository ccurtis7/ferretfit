import diff_classifier.aws as aws
import pandas as pd
import boto3
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.optimize import leastsq
from scipy.stats import sem

def intersection(x1,y1,x2,y2):
    """
    Adapted from sukhbinder at 
    https://github.com/sukhbinder/intersection/blob/master/intersection.py
    
    INTERSECTIONS Intersections of curves.
   Computes the (x,y) locations where two curves intersect.  The curves
   can be broken with NaNs or have vertical segments.

    usage:
    x,y=intersection(x1,y1,x2,y2)

    Example:
    a, b = 1, 2
    phi = np.linspace(3, 10, 100)
    x1 = a*phi - b*np.sin(phi)
    y1 = a - b*np.cos(phi)

    x2=phi
    y2=np.sin(phi)+2
    x,y=intersection(x1,y1,x2,y2)

    plt.plot(x1,y1,c='r')
    plt.plot(x2,y2,c='g')
    plt.plot(x,y,'*k')
    plt.show()

    """
    ii,jj=_rectangle_intersection_(x1,y1,x2,y2)
    n=len(ii)

    dxy1=np.diff(np.c_[x1,y1],axis=0)
    dxy2=np.diff(np.c_[x2,y2],axis=0)

    T=np.zeros((4,n))
    AA=np.zeros((4,4,n))
    AA[0:2,2,:]=-1
    AA[2:4,3,:]=-1
    AA[0::2,0,:]=dxy1[ii,:].T
    AA[1::2,1,:]=dxy2[jj,:].T

    BB=np.zeros((4,n))
    BB[0,:]=-x1[ii].ravel()
    BB[1,:]=-x2[jj].ravel()
    BB[2,:]=-y1[ii].ravel()
    BB[3,:]=-y2[jj].ravel()

    for i in range(n):
        try:
            T[:,i]=np.linalg.solve(AA[:,:,i],BB[:,i])
        except:
            T[:,i]=np.NaN


    in_range= (T[0,:] >=0) & (T[1,:] >=0) & (T[0,:] <=1) & (T[1,:] <=1)

    xy0=T[2:,in_range]
    xy0=xy0.T
    return xy0[:,0],xy0[:,1]


def _rect_inter_inner(x1,x2):
    n1=x1.shape[0]-1
    n2=x2.shape[0]-1
    X1=np.c_[x1[:-1],x1[1:]]
    X2=np.c_[x2[:-1],x2[1:]]
    S1=np.tile(X1.min(axis=1),(n2,1)).T
    S2=np.tile(X2.max(axis=1),(n1,1))
    S3=np.tile(X1.max(axis=1),(n2,1)).T
    S4=np.tile(X2.min(axis=1),(n1,1))
    return S1,S2,S3,S4


def _rectangle_intersection_(x1,y1,x2,y2):
    S1,S2,S3,S4=_rect_inter_inner(x1,x2)
    S5,S6,S7,S8=_rect_inter_inner(y1,y2)

    C1=np.less_equal(S1,S2)
    C2=np.greater_equal(S3,S4)
    C3=np.less_equal(S5,S6)
    C4=np.greater_equal(S7,S8)

    ii,jj=np.nonzero(C1 & C2 & C3 & C4)
    return ii,jj


def fit_sine(x, y, guess_freq=1/150):

    guess_mean = np.mean(y)
    guess_std = 3*np.std(y)/(2**0.5)/(2**0.5)
    guess_slope = 0
    if x.shape[0] > 8:
        guess_freq = 1/(0.7*np.average(np.diff(x)))
        guess_amp = 2*(min(y) - max(y))
    else:
        guess_freq = 1/(0.5*np.average(np.diff(x)))
        guess_amp = 0.75*np.ptp(y)
        
    guess_phase = np.arcsin((y[0] - guess_mean)/guess_amp) - x[0]
    
    # we'll use this to plot our first estimate. This might already be good enough for you
    data_first_guess = guess_std*np.sin(x+guess_phase) + guess_mean + guess_slope*x

    # Define the function to optimize, in this case, we want to minimize the difference
    # between the actual data and our "guessed" parameters
    if x.shape[0] > 8:
        optimize_func = lambda z: z[0]*np.sin(z[1]*x+z[2]) + z[3] +z[4]*x - y
        est_amp, est_freq, est_phase, est_mean, est_slope = leastsq(optimize_func,
                                                                [guess_amp, guess_freq,
                                                                 guess_phase, guess_mean, guess_slope])[0]

    else:
        optimize_func = lambda z: guess_amp*np.sin(z[0]*x+z[1]) + z[2] +z[3]*x - y
        est_freq, est_phase, est_mean, est_slope = leastsq(optimize_func,
                                                                [guess_freq,
                                                                 guess_phase, guess_mean, guess_slope])[0]
        est_amp, est_freq, est_phase, est_mean, est_slope = guess_amp, guess_freq, guess_phase, guess_mean, guess_slope

    # recreate the fitted curve using the optimized parameters
    data_fit = est_amp*np.sin(est_freq*x+est_phase) + est_mean + est_slope*x

    # recreate the fitted curve using the optimized parameters

    fine_t = np.arange(0,max(x),0.1)
    data_fit=est_amp*np.sin(est_freq*fine_t+est_phase)+est_mean+est_slope*fine_t
    
    return fine_t, data_fit, est_amp, est_freq


def ferret_fit(folder, prefix, download=True, bucket_name='ccurtis.data'):
    
    filename = '{}.csv'.format(prefix)
    if download:
        aws.download_s3('{}/{}'.format(folder, filename), filename, bucket_name=bucket_name)
    ferret_data = pd.read_csv(filename)
    ferret_data = ferret_data.sort_values(by=['X'])
    length = ferret_data.shape[0]

    x = ferret_data['X']
    y = ferret_data['Y']
    fine_t, data_fit, est_amp, est_freq = fit_sine(x, y, guess_freq=1/100)

    lowess = sm.nonparametric.lowess
    ymid = lowess(y, x, frac=0.3)
    yavg = np.convolve(y, np.ones((length,))/length, mode='same')

    strait = np.mean(y)*np.ones((length,))
    intersections = intersection(x, ymid[:,1], x, strait)
    pawcount = len(x)
    pawdens = np.abs(100*pawcount/(max(x) - min(x)))
    stride = np.mean(np.diff(x))
    stridestd = np.std(np.diff(x))
    
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, s=300)
    plt.plot(x, ymid[:, 1], linewidth=6)
    plt.plot(x, strait, 'k', linewidth=6)
    plt.plot(fine_t, data_fit, 'm', linewidth=6)
    #plt.plot(x, yavg, 'k', linewidth=6)
    plt.ylim(0, 120)
    imfile = '{}_fit.png'.format(prefix)
    plt.savefig(imfile)
    aws.upload_s3(imfile, '{}/{}'.format(folder, imfile), bucket_name='ccurtis.data')
    
    ystd = np.round(np.std(y), 2)
    yrange = np.round(np.ptp(y), 2)
    rsd = 100*np.round(ystd/np.mean(y), 2)
    cross = len(intersections[0])
    crossdens = np.abs(100*cross/(max(x) - min(x)))

    print('Video to analyze: {}'.format(filename))
    print('Deviation from midline: {}'.format(ystd))
    print('Range in y: {}'.format(yrange))
    print('Percent deviation from midline: {}'.format(rsd))
    print('Fit amplitude: {}'.format(np.abs(np.round(est_amp, 2))))
    print('Number of intersections: {}'.format(cross))
    print('Number of intersections per 100 pixels: {}'.format(np.round(crossdens)))
    print('Number of footprints: {}'.format(pawcount))
    print('Number of footprints per 100 pixels: {}'.format(np.round(pawdens)))
    print('Average stride: {}'.format(np.round(stride)))
    print('Stride deviation: {}'.format(np.round(stridestd)))
    print('Fit period: {}\n'.format(np.round(1/est_freq, 2)))
    
    
    return (ystd, yrange, rsd, np.abs(est_amp), 1/est_freq, pawcount, pawdens, cross, crossdens, stride, stridestd)
