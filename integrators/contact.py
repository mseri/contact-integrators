r"""
Contact integrators defined for an Hamiltonian of the form
$$
  H(q, p, s) = \frac{p^2}2 + f(t)s + V(q,t),
$$
or, equivalently, a Lagrangian of the form
$$
  L(q, \dot{q}, s) = \frac{\dot{q}^2}2 - f(t)s - V(q,t).
$$

These systems are provided via a class providing the functions

- V(q,t) -- the potential (this can include a forcing term),
- Vq(q,t) -- d/dq V(q,t),
- f(t) -- the time-dependent damping.
"""

import numpy as np
import warnings


def variational_step(system, dt, p, x, s, t):
    # TODO: add snew = s + dt * L
    xnew = x + (dt - 0.5 * dt**2 * system.f(t)) * \
        p - 0.5 * dt**2 * system.Vq(x, t)
    pnew = (1.0-0.5*dt*system.f(t))/(1.0 + 0.5*dt*system.f(t))*p + 0.5*dt*(
        - system.Vq(x, t) - system.Vq(xnew, t)
    )/(1.0 + 0.5*dt*system.f(t))
    tnew = t + dt
    return pnew, xnew, 0, tnew


def symcontact(system, tspan, p0, q0):
    """
    Integrate [system] with initial conditions [p0], [q0]
    using the second order contact variational integrator.

    [tspan] is usually [np.linspace(t0, tfinal, num=steps)]
    """
    dt = tspan[1]-tspan[0]
    steps = len(tspan)
    init = [p0, q0]

    sol = np.empty([steps, *np.shape(init)], dtype=np.float64)
    sol[0] = np.array(init)

    for i, t in enumerate(tspan):
        p, x = sol[i]
        pnew, xnew, _, _ = variational_step(system, dt, p, x, 0, t)
        sol[i+1] = np.array((pnew, xnew))
    return sol, tspan


def step(system, dt, p, q, s, t):
    # dt/2 D
    t += dt/2.

    # dt/2 C
    q += p*dt/2.
    s += np.linalg.norm(p)**2*dt/4.

    # dt/2 B
    p -= system.Vq(q, t)*dt/2
    s -= system.V(q, t)*dt/2

    # dt A
    etf = np.exp(-dt*system.f(t))
    p *= etf
    s *= etf

    # dt/2 B
    p -= system.Vq(q, t)*dt/2
    s -= system.V(q, t)*dt/2

    # dt/2 C
    q += p*dt/2
    s += np.linalg.norm(p)**2*dt/4

    # dt/2 D
    t += dt/2

    return (p, q, s, t)


_A = [-1.17767998417887, 0.235573213359357, 0.784513610477560]
_A0 = 1-2*sum(_A)
a_six = _A[::-1] + [_A0] + _A[:]

_B = [-2.13228522200144, 0.00426068187079180, 1.43984816797678]
_B0 = 1-2*sum(_B)
b_six = _B[::-1] + [_B0] + _B[:]

_C = [0.00152886228424922, -2.14403531630539, 1.44778256239930]
_C0 = 1-2*sum(_C)
c_six = _C[::-1] + [_C0] + _C[:]

x0, x1 = np.array([- np.cbrt(2), 1])/(2 - np.cbrt(2))
z0, z1 = np.array([- np.power(2, 1/5), 1])/(2 - np.power(2, 1/5))
_E = [x0*z0, x1*z0, x1*z1, x0*z1, x1*z1]
e_six = _E[-1:0:-1] + _E[:]


def step6(system, dt, p, q, s, t, a=a_six, stepper=step):
    for coeff in a:
        p, q, s, t = stepper(system, coeff*dt, p, q, s, t)
    return p, q, s, t


def step6e(system, dt, p, q, s, t): return step6(
    system, dt, p, q, s, t, a=e_six)


def step6b(system, dt, p, q, s, t): return step6(
    system, dt, p, q, s, t, a=b_six)


def step6c(system, dt, p, q, s, t): return step6(
    system, dt, p, q, s, t, a=c_six)


def integrate(stepper, system, tspan, p0, q0, s0, ttol=1e-13):
    """
    Integrate [system] with initial conditions [p0], [q0]
    using the hamiltonian integrator provided in step.

    [tspan] is usually [np.linspace(t0, tfinal, num=steps)]
    """
    dt = tspan[1] - tspan[0]
    steps = len(tspan)
    init = [p0, q0]

    solpq = np.empty([steps, *np.shape(init)], dtype=np.float64)
    sols = np.empty(steps, dtype=np.float64)
    solpq[0] = np.array(init)
    sols[0] = s0

    for i in range(steps-1):
        p, q = solpq[i]
        s = sols[i]
        t = tspan[i]
        pnew, qnew, snew, tnew = stepper(system, dt, p, q, s, t)
        if abs(tnew-t-dt) > ttol:
            warnings.warn(f"tnew-t-dt, dt inconsistency: {tnew-t-dt}, {dt}")
        solpq[i+1] = [pnew, qnew]
        sols[i+1] = snew

    return solpq, sols, tspan
