import numpy as np
import numpy.testing as npt
import pandas as pd
import ferretfit.ferretfit as ff


def test_intersection():
    # test 1
    a, b = 1, 2
    phi = np.linspace(3, 10, 100)
    x1 = a*phi - b*np.sin(phi)
    y1 = a - b*np.cos(phi)

    x2 = phi
    y2 = np.sin(phi)+2
    x, y = ff.intersection(x1, y1, x2, y2)
    assert np.round(np.average(x), 1) == 7.2
    assert np.round(np.average(y), 1) == 2.3

    # test 2
    x1 = np.linspace(0, 10, 100)
    y1 = -x1+5
    y2 = x1
    x, y = ff.intersection(x1, y1, x1, y2)
    assert np.round(x, 1)[0] == 2.5
    assert np.round(y, 1)[0] == 2.5

    # test 3
    x1 = np.linspace(0, 10, 100)
    y1 = x1+5
    y2 = x1
    x, y = ff.intersection(x1, y1, x1, y2)
    assert x.shape[0] == 0


def test_rect_inter_inner():
    x1 = np.linspace(0, 10, 3)
    x2 = np.linspace(20, 30, 3)
    S1, S2, S3, S4 = ff.rect_inter_inner(x1, x2)
    assert np.mean(S1) == 2.5
    assert S1.shape == (2, 2)


def test_rectangle_intersection():
    x1 = np.linspace(0, 10, 3)
    y1 = -x1+6
    y2 = x1
    x, y = ff.rectangle_intersection(x1, y1, x1, y2)
    assert x.tolist() == [0, 0, 1]
    assert y.tolist() == [0, 1, 0]


def test_():
