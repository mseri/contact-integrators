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