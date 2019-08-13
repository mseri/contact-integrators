import numpy as np
import matplotlib.pylab as plb


def rk4(system, init, tspan):
    """
    RungeKutta4 from matplotlib.pylab for the damped oscillator
    with damping factor a.
    """

    def derivs(x, t):
        dq = x[:len(x)//2]
        dp = -system.Vq(x[len(x)//2:], t) - system.f(t)*x[:len(x)//2]
        return tuple(np.concatenate([dp, dq]))
    return plb.rk4(derivs, init, tspan)


# From https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def pad_and_cumsum(vec):
    cumsum = np.cumsum(vec)
    return shift(cumsum, 1, 0.0)
