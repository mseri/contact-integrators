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


_A = [1.315186320683906, -1.17767998417887,
      0.235573213359357, 0.784513610477560]
a_six = _A[-1:0:-1] + [_A[0]] + _A[1:]


def step6(system, dt, p, q, s, t):
    for coeff in a_six:
        p, q, s, t = step(system, coeff*dt, p, q, s, t)
    return p, q, s, t


def integrate(stepper, system, tspan, p0, q0, s0):
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
        pnew, qnew, snew, _ = stepper(system, dt, p, q, s, t)
        solpq[i+1] = [pnew, qnew]
        sols[i+1] = snew

    return solpq, sols, tspan
