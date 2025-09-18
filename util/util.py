"""
	Utility Toolbox

	Written by Dalen Industries <00215069@protonmail.com>, June 2020
        C.D.
"""
import math
import numpy as np
from scipy import linalg as la
import control as con

def ss2cf(a, b, d):
    """
	Call:
        ac, bc, dc, t = ss2cf(a,b,d)
    Purpose
    Transform a state space model (a,b,d) into observable canonical form
    (ac,bc,dc).
    xh = a x + b u     -->          xch = ac xc + bc u
    y  = d x + e u     -->          y   = dc xc + e  u
    Note:
    The matrix, e, is not influenced by state coordinate transformation.
    """
    n, _ = np.shape(b)
    o = d
    for i in range(1, n):
        o = np.vstack((o, d@a**i))
    t = o[:n, :]
    oc = o@la.inv(t)
    dc = d@la.inv(t)
    ac = la.pinv(oc)@o@a@la.inv(t)
    bc = la.pinv(oc)@o@b
    return ac, bc, dc, t

def prbs1(N, Tmin, Tmax):
    """
    PRBS1
	Call:
        u, t = prbs1(N,Tmin,Tmax)
    Purpose:
    Make a Pseudo Random Binary Signal of lenght, N, samples.
    The signal is constant for a random interval of, T, samples.
    The random interval, T, is bounded by a specified band, Tmax <= T <= Tmax.
    N - Number of samples in input signal, u, (u_t, for all t=1,...,N).
    Tmin - Minimal interval for which, u_t, is constant.
    Tmax - Maximal interval for which, u_t, is constant.
    u - The PRBS input signal of lenght, N, samples.
    ALGORITHM:
    Simply a home made algorithm. DDiR, revised 17.2.2000
    """
    is2 = 1
    Tmin = Tmin-1
    dT = Tmax-Tmin
    u = np.zeros([2*N, 1]); t = np.zeros([2*N, 1])
    if is2 == 0:
        s = np.sign(np.random.randn())
    else:
        s = 1
        k = 1
    while k < N+1:
        T = Tmin+dT*np.random.rand()
        T = int(np.ceil(T))
        u[k-1:k+T-1] = s*np.ones([T, 1])
        s = s*(-1)
        t[k] = T
        k = k+T
    u = u[:N]
    return u, t

def rctrb2(A, B, L):
    """
    RCTRB2 Reversed controllability matrix
           Return the reversed extended controllability matrix
           C_L=[A^{L-1}B ... AB B]
    Call
        C_L = rctrb2(A,B,L)
    Input:
        A,B  System matrices.
        L    # of block columns in C_L.
    Output:
        C_L  The reversed extended controlability matrix.
    """
    n, nu = np.shape(B)
    C = np.zeros(n, L*nu)
    w = B
    C[:, (L-1)*nu:L*nu] = w
    for i in range(2, L+1):
        w = A*w
        j = (L-i)*nu
        C[:, j:j+nu] = w
    return C

def d2c(dsys):
    """
    D2C Converts discrete time dynamic system to continuous time.
    Call:
        csys = d2c(dsys)
    computes a continuous time model CSYS that
    approximates the discrete time model DSYS.
    Algorithm: Zero-order hold
    """
    a = dsys.A; b = dsys.B
    c = dsys.C; d = dsys.D
    dt = dsys.dt
    n, r = np.shape(b)
    WORK1 = np.hstack((a, b))
    WORK2 = np.hstack((np.zeros((r, n)), np.eye(r)))
    WORK3 = np.vstack((WORK1, WORK2))
    s = la.logm(WORK3)
    s = s/dt
    s = np.real(s)
    A = s[:n, :n]; B = s[:n, n:n+r]
    csys = con.StateSpace(A, B, c, d)
    return csys

def sim(dsys, Kp, Ti, Td, T, nt=0, r=1):
    """
    sim Simulation of a closed loop discrete time linear systems.
    Call:
        sim(dsys, Kp, Ti, Td, T, nt=0, r=1)
        sim(dsys, Kp, Ti, Td, T, nt=0)
        sim(dsys, Kp, Ti, Td, T)
    Input:
        dsys - Discrete dynamic model
        Kp - The proportional constant
        Ti - The Integral time constant
        T - Time Vector, (N x 1), N is len(T)
        #x - Initial state vector, (n x 1), n is # states
        nt - sample delay
    Output:
        Y - Matrix, Y, with system outputs, (N x 1), m is # outputs
        U - Matrix, U, with system inputs, (N x 1), r is # inputs
    """
    a = dsys.A; b = dsys.B
    d = dsys.C
    dt = dsys.dt
    N = len(T)
    KpTih = Kp*dt/Ti; KpTdh = Kp*Td/dt
    Y = np.zeros([N, 1]); U = np.zeros([N, 1]); yt = np.zeros([nt, 1])
    z = 0; y_old = 0; x = np.zeros([np.shape(a)[0], 1])
    y_old = d*x
    for k in range(0, N):
        if nt == 0:
            ym = d*x
        else:
            y = d*x
            ym = yt[nt-1]
            for i in range(nt, 1, -1):
                yt[i-1] = yt[i-2]
            yt[0] = y
        e = r-ym
        u = Kp*e+z-KpTdh*(ym-y_old)
        z = z+KpTih*e
        Y[k] = ym
        U[k] = u
        y_old = y
        x = a*x+b*u
    return Y, U

def d2d(dsys, dt2):
    """
    d2d Resamples discrete time dynamic system.
    Call:
        dsys2 = d2d(dsys, dt2)
    resamples the discrete time dynamic system
    to target sample time dt2.
    """
    a = dsys.A; b = dsys.B
    c = dsys.C; d = dsys.D
    dt = dsys.dt
    n, r = np.shape(b)
    WORK1 = np.hstack((a, b))
    WORK2 = np.hstack((np.zeros((r, n)), np.eye(r)))
    WORK3 = np.vstack((WORK1, WORK2))
    p = dt2/dt
    s = la.fractional_matrix_power(WORK3, p)
    A = s[:n, :n]; B = s[:n, n:n+r]
    dsys2 = con.StateSpace(A, B, c, d, dt2)
    return dsys2

def dcgain(dsys):
    """
    dcgain dc gain of dynamic systems.
    Call:
        gain = dcgain(dsys)
    computes the steady state gain
    of the dynamic system dsys.
    """
    csys = d2c(dsys)
    gain = -csys.C*la.pinv(csys.A)*csys.B+csys.D
    return gain

def dlqdu_pi(A,B,D,Q,Rw):
    nx = np.shape(A)[0]; nu=np.shape(B)[1]; ny=np.shape(D)[0]
    Dt = np.hstack([D,np.eye(ny,ny,dtype=float)])
    At = np.vstack([np.hstack([A,np.zeros([nx,ny])]),Dt])
    Bt = np.vstack([B,np.zeros([ny,nu])])
    Qt = Dt.T*Q
    Qt = Qt*Dt
    P = np.matrix(la.solve_discrete_are(At, Bt, Qt, Rw))
    K = np.matrix(la.inv(Bt.T*P*Bt+Rw)*(Bt.T*P*At))
    G = -K
    G1 = G[:,0:nx]; G2=G[:,nx:nx+ny]
    return G1, G2

