'''
constrained HF with given value of lambda
author: Chenghan Li
'''

from pyscf.scf.hf import RHF
import numpy as np

class CRHF(RHF):
    def __init__(self, mol, bondatmlst, lmbda, *args, **kwargs):
        super().__init__(mol, *args, **kwargs)
        self.bondatmlst = bondatmlst
        self.lmbda = lmbda
        assert len(self.bondatmlst) == len(self.lmbda)

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        vhf = \
        super().get_veff(mol=mol, dm=dm, dm_last=dm_last, vhf_last=vhf_last, hermi=hermi)
        return vhf + self.get_lambda(self.get_ovlp(), dm)

    def get_lambda(self, s1e, dm):
        L = np.zeros_like(dm)
        dmS = dm @ s1e
        aoslices = self.mol.aoslice_by_atom()
        for bond, l in zip(self.bondatmlst, self.lmbda):
            a0, a1 = aoslices[bond[0], 2:]
            b0, b1 = aoslices[bond[1], 2:]
            tmp = l * s1e[:,b0:b1] @ dmS[b0:b1,a0:a1] / 2
            L[:,a0:a1] -= tmp 
            L[a0:a1]   -= tmp.T
            tmp = l * s1e[:,a0:a1] @ dmS[a0:a1,b0:b1] / 2
            L[:,b0:b1] -= tmp
            L[b0:b1]   -= tmp.T
        return L

    def bond_order(self, s1e, dm):
        bo = list()
        dmS = dm @ s1e
        aoslices = self.mol.aoslice_by_atom()
        for bond in self.bondatmlst:
            a0, a1 = aoslices[bond[0], 2:]
            b0, b1 = aoslices[bond[1], 2:]
            bo.append(np.trace(dmS[a0:a1,b0:b1] @ dmS[b0:b1,a0:a1]))
        return bo

    def get_lambda_numerical(self, s1e, dm, dx=1e-4):
        L = np.zeros_like(dm)
        for i in range(dm.shape[0]):
            for j in range(dm.shape[1]):
                dm_p = dm.copy()
                dm_p[i,j] += dx / 2
                bo_p = self.bond_order(s1e, dm_p)
                dm_m = dm.copy()
                dm_m[i,j] -= dx / 2
                bo_m = self.bond_order(s1e, dm_m)
                for bp, bm, l in zip(bo_p, bo_m, self.lmbda):
                    L[i,j] -= (bp - bm) * l / dx
        return (L + L.T) / 2

    def kernel(self, conv_tol=1e-10, conv_tol_grad=None,
           dump_chk=True, dm0=None, callback=None, conv_check=True, **kwargs):
        pass

if __name__ == "__main__":
    from pyscf import gto, scf
    
    mol = gto.Mole()
    mol.verbose = 5
    mol.atom = '''
    O        0.000000    0.000000    0.117790
    H        0.000000    0.755453   -0.471161
    H        0.000000   -0.755453   -0.471161'''
    mol.basis = 'ccpvdz'
    mol.build()
    
    from pyscf.constrained.scf.hf import CRHF
    mf = scf.RHF(mol)
    mf.kernel()
    dm0 = mf.make_rdm1()
    
    mf = CRHF(mol, [[0,1]], [-1e-1])
    mf.max_cycle = 100
    mf.kernel(dm0=dm0)
    
    s1e = mf.get_ovlp()
    print("lambda = 0, bond order =", mf.bond_order(s1e, dm0))
    print("lambda = -0.1, bond order =", mf.bond_order(s1e, mf.make_rdm1()))
