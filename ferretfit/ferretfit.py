"""Functions used to calculate parameters related to lateral spread in Catwalk
experiments

"""


import boto3
import pandas as pd
import numpy as np
from scipy.optimize import leastsq
from scipy.stats import sem
import matplotlib.pyplot as plt
import statsmodels.api as sm
import ferretfit.aws as aws


class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def intersection(x1, y1, x2, y2):
    """Finds the intersection of two curves

    Parameters
    ----------
    x1 : numpy.ndarray
        x coordinates of first curve
    y1 : numpy.ndarray
        y coordinates of first curve
    x2 : numpy.ndarray
        x coordinates of second curve
    y2 : numpy.ndarray
        y coordinates of second curve

    Returns
    -------
    xy0[:, 0] : numpy.ndarray
        x coordinates of all intersection points
    xy0[:, 1] : numpy.ndarray
        y coordinates of all intersection points

    Examples
    --------
    >>> x1 = np.linspace(0, 10, 100)
    >>> y1 = -x1+5
    >>> y2 = x1
    >>> ff.intersection(x1, y1, x1, y2)
    (array([2.5]), array([2.5]))

    Notes
    -----
    Adapted from sukhbinder at
    https://github.com/sukhbinder/intersection/blob/master/intersection.py

    INTERSECTIONS Intersections of curves.
    Computes the (x,y) locations where two curves intersect.  The curves
    can be broken with NaNs or have vertical segments.

    """
    ii, jj = rectangle_intersection(x1, y1, x2, y2)
    n = len(ii)

    dxy1 = np.diff(np.c_[x1, y1], axis=0)
    dxy2 = np.diff(np.c_[x2, y2], axis=0)

    T = np.zeros((4, n))
    AA = np.zeros((4, 4, n))
    AA[0:2, 2, :] = -1
    AA[2:4, 3, :] = -1
    AA[0::2, 0, :] = dxy1[ii, :].T
    AA[1::2, 1, :] = dxy2[jj, :].T

    BB = np.zeros((4, n))
    BB[0, :] = -x1[ii].ravel()
    BB[1, :] = -x2[jj].ravel()
    BB[2, :] = -y1[ii].ravel()
    BB[3, :] = -y2[jj].ravel()

    for i in range(n):
        try:
            T[:, i] = np.linalg.solve(AA[:, :, i], BB[:, i])
        except:
            T[:, i] = np.NaN

    in_range = (T[0, :] >= 0) & (T[1, :] >= 0) & (T[0, :] <= 1) & (T[1, :] <= 1)

    xy0 = T[2:, in_range]
    xy0 = xy0.T
    return xy0[:, 0], xy0[:, 1]


def rect_inter_inner(x1, x2):
    """Constructs coordinate pairs of rectangles from input arrays

    Creates 4 new arrays holding lower x, upper y, upper x, and lower y
    coordinates from input (x, y) coordinate pairs (corresponding to x1, y1
    inputs, respectively). Creates an 4 n - 1 x m - 1 arrays.

    Parameters
    ----------
    x1 : numpy.ndarray
        First input array
    x2 : numpy.ndarray
        Second input array

    Returns
    -------
    S1 : numpy.ndarray
        Array holding lower x coordinates
    S2 : numpy.ndarray
        Array holding upper y coordinates
    S3 : numpy.ndarray
        Array holding upper x coordinates
    S4 : numpy.ndarray
        Array holding lower y coordinates

    Examples
    --------
    >>> x1 = np.linspace(0, 10, 3)
    >>> x2 = np.linspace(20, 30, 3)
    >>> ff.rect_inter_inner(x1, x2)
    (array([[0., 0.],
    [5., 5.]]), array([[25., 30.],
    [25., 30.]]), array([[ 5.,  5.],
    [10., 10.]]), array([[20., 25.],
    [20., 25.]]))

    """

    n1 = x1.shape[0]-1
    n2 = x2.shape[0]-1
    X1 = np.c_[x1[:-1], x1[1:]]
    X2 = np.c_[x2[:-1], x2[1:]]
    S1 = np.tile(X1.min(axis=1), (n2, 1)).T
    S2 = np.tile(X2.max(axis=1), (n1, 1))
    S3 = np.tile(X1.max(axis=1), (n2, 1)).T
    S4 = np.tile(X2.min(axis=1), (n1, 1))
    return S1, S2, S3, S4


def rectangle_intersection(x1, y1, x2, y2):
    """Intermediate function for calculating intersection of two curves

    Parameters
    ----------
    x1 : numpy.ndarray
        x coordinates of first curve
    y1 : numpy.ndarray
        y coordinates of first curve
    x2 : numpy.ndarray
        x coordinates of second curve
    y2 : numpy.ndarray
        y coordinates of second curve

    Returns
    -------
    ii : numpy.ndarray
        x coordinates of intersecting rectangles
    jj : numpy.ndarray
        y coordinates of intersecting rectangles

    Examples
    --------
    >>> x1 = np.linspace(0, 10, 3)
    >>> y1=-x1+6
    >>> y2=x1
    >>> ff.rectangle_intersection(x1, y1, x1, y2)
    (array([0, 0, 1], dtype=int64), array([0, 1, 0], dtype=int64))

    """
    S1, S2, S3, S4 = rect_inter_inner(x1, x2)
    S5, S6, S7, S8 = rect_inter_inner(y1, y2)

    C1 = np.less_equal(S1, S2)
    C2 = np.greater_equal(S3, S4)
    C3 = np.less_equal(S5, S6)
    C4 = np.greater_equal(S7, S8)

    ii, jj = np.nonzero(C1 & C2 & C3 & C4)
    return ii, jj


def fit_sine(x, y, gfr=1/150):
    """An optimized function for fitting CatWalk data to sinusoidal curve.

    Parameters
    ----------
    x : numpy.ndarray
        x coordinates of pawprints
    y : numpy.ndarray
        y coordinates of pawprints
    gfr : int or float
        An initial guess of the frequency of the fitted sine curve

    Returns
    -------
    fine_t : numpy.ndarray
        x coordinates of fitted sine curve
    data_fit : numpy.ndarray
        y coordinates of fitted sine curve
    eamp : float
        Amplitude of fitted sine curve
    efr : float
        Frequency of fitted sine curve

    Examples
    --------


    """

    gmean = np.mean(y)
    guess_std = 3*np.std(y)/(2**0.5)/(2**0.5)
    gslp = 0
    if x.shape[0] > 8:
        gfr = 1/(0.7*np.average(np.diff(x)))
        gamp = 2*(min(y) - max(y))
    else:
        gfr = 1/(0.5*np.average(np.diff(x)))
        gamp = 0.75*np.ptp(y)

    gphas = np.arcsin((y[0] - gmean)/gamp) - x[0]

    # we'll use this to plot our first estimate.
    # This might already be good enough for you
    data_first_guess = guess_std*np.sin(x+gphas) + gmean + gslp*x

    # Define the function to optimize, in this case,
    # we want to minimize the difference
    # between the actual data and our "guessed" parameters
    if x.shape[0] > 8:
        optimize_func = lambda z: z[0]*np.sin(z[1]*x+z[2]) + z[3] + z[4]*x - y
        eamp, efr, ephas, em, eslp = leastsq(optimize_func, [gamp, gfr, gphas,
                                                             gmean, gslp])[0]

    else:
        optimize_func = lambda z: gamp*np.sin(z[0]*x+z[1]) + z[2] + z[3]*x - y
        efr, ephas, em, eslp = leastsq(optimize_func, [gfr, gphas,
                                                       gmean, gslp])[0]
        eamp, efr, ephas, em, eslp = gamp, gfr, gphas, gmean, gslp

    # recreate the fitted curve using the optimized parameters
    data_fit = eamp*np.sin(efr*x+ephas) + em + eslp*x

    # recreate the fitted curve using the optimized parameters

    fine_t = np.arange(0, max(x), 0.1)
    data_fit = eamp*np.sin(efr*fine_t+ephas)+em+eslp*fine_t

    return fine_t, data_fit, eamp, efr


def ferret_fit(folder, prefix, download=True, bucket_name='ccurtis.data'):

    filename = '{}.csv'.format(prefix)
    if download:
        aws.download_s3('{}/{}'.format(folder, filename), filename,
                        bucket_name=bucket_name)
    ferret_data = pd.read_csv(filename)
    ferret_data = ferret_data.sort_values(by=['X'])
    length = ferret_data.shape[0]

    x = ferret_data['X']
    y = ferret_data['Y']
    fine_t, data_fit, eamp, efr = fit_sine(x, y, gfr=1/100)

    lowess = sm.nonparametric.lowess
    ymid = lowess(y, x, frac=0.3)
    yavg = np.convolve(y, np.ones((length,))/length, mode='same')

    strait = np.mean(y)*np.ones((length,))
    intersections = intersection(x, ymid[:, 1], x, strait)
    pawcount = len(x)
    pawdens = np.abs(100*pawcount/(max(x) - min(x)))
    stride = np.mean(np.diff(x))
    stridestd = np.std(np.diff(x))

    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, s=300)
    plt.plot(x, ymid[:, 1], linewidth=6)
    plt.plot(x, strait, 'k', linewidth=6)
    plt.plot(fine_t, data_fit, 'm', linewidth=6)
    # plt.plot(x, yavg, 'k', linewidth=6)
    plt.ylim(0, 120)
    imfile = '{}_fit.png'.format(prefix)
    plt.savefig(imfile)
    aws.upload_s3(imfile, '{}/{}'.format(folder, imfile),
                  bucket_name='ccurtis.data')

    ystd = np.round(np.std(y), 2)
    yrange = np.round(np.ptp(y), 2)
    rsd = 100*np.round(ystd/np.mean(y), 2)
    cross = len(intersections[0])
    crossdens = np.abs(100*cross/(max(x) - min(x)))

    ffparams = Bunch(ystd=ystd, yrange=yrange, rsd=rsd, amp=np.abs(eamp),
                     period=1/efr, pawcount=pawcount, pawdens=pawdens,
                     cross=cross, crossdens=crossdens, stride=strid,
                     stridestd=stridestd)

    print('Video to analyze: {}'.format(filename))
    print('Deviation from midline: {}'.format(ystd))
    print('Range in y: {}'.format(yrange))
    print('Percent deviation from midline: {}'.format(rsd))
    print('Fit amplitude: {}'.format(np.abs(np.round(eamp, 2))))
    print('Number of intersections: {}'.format(cross))
    print('Number of intersections per 100 pixels: {}'.format(np.round(crossdens
                                                                       )))
    print('Number of footprints: {}'.format(pawcount))
    print('Number of footprints per 100 pixels: {}'.format(np.round(pawdens)))
    print('Average stride: {}'.format(np.round(stride)))
    print('Stride deviation: {}'.format(np.round(stridestd)))
    print('Fit period: {}\n'.format(np.round(1/efr, 2)))

    return ffparams
