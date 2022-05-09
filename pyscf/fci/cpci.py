from pyscf.fci import direct_spin1 
from pyscf.fci import rdm as fci_rdm

import scipy.sparse.linalg as sla
import numpy as np

def solve(h1e, eri, ECI, ci, dHc, norb, nelec, 
        dci0=None, tol=1e-6, max_cycle=20):
    '''
    solve for dc^CI / dR from 
    (H^CI - E^CI + 2 c^CI \outer c^CI) dc^CI / dR = dHc
    i.e. M dc = dHc

    the solution is c^CI response if dHc = d(E^CI-H^CI) / dR \dot c^CI

    h1e: 1e integral
    eri: 2e integral
    ECI: CI energy
    ci:  CI vector
    '''
    lidxa, lidxb = direct_spin1._unpack(norb, nelec, None)
    Na = lidxa.shape[0]; Nb = lidxb.shape[0]

    h2e = direct_spin1.absorb_h1e(h1e, eri, norb, nelec, 0.5)

    def M(dc):
        '''
        NOTE dc will be a flattened vector with shape Na*Nb
        '''
        cdc = np.dot(ci.reshape(-1), dc)
        Mdc = 2 * (cdc * ci)
        dc = dc.reshape((Na, Nb))
        Mdc += direct_spin1.contract_2e(h2e, dc, norb, nelec, (lidxa,lidxb))
        Mdc -= (ECI * dc)
        return Mdc

    M = sla.LinearOperator((Na*Nb,Na*Nb), M)

    if dci0 is None:
        dci0 = np.zeros(Na*Nb)
    sol, stat = sla.gmres(M, dHc.flatten(), x0=dci0,
            tol=tol, atol=tol*np.linalg.norm(dHC), maxiter=max_cycle)
    # stat == 0 means converged

    return sol.reshape((Na,Nb)), stat

