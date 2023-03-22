'''
constrained HF with given value of lambda
author: Chenghan Li
'''

from pyscf.scf.hf import RHF
import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import root

class CRHF(RHF):
    def __init__(self, mol, bondatmlst, bond_order_constraint, *args, **kwargs):
        super().__init__(mol, *args, **kwargs)
        self.bondatmlst = bondatmlst
        self.bond_order_constraint = bond_order_constraint
        self.lagrange_multiplier = np.zeros(len(bond_order_constraint))
        self.original_veff = None
        assert len(self.bondatmlst) == len(self.bond_order_constraint)

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        vhf = \
        super().get_veff(mol=mol, dm=dm, dm_last=dm_last, vhf_last=vhf_last, hermi=hermi)
        self.original_veff = vhf
        return vhf + self.get_lambda(self.get_ovlp(), dm)

    def get_lambda(self, s1e, dm):
        L = np.zeros_like(dm)
        dmS = dm @ s1e
        aoslices = self.mol.aoslice_by_atom()
        for bond, l, constraint in zip(self.bondatmlst, self.lagrange_multiplier, self.bond_order_constraint):
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
                for bp, bm, l in zip(bo_p, bo_m, self.lagrange_multiplier):
                    L[i,j] -= (bp - bm) * l / dx
        return (L + L.T) / 2

    def kernel(self, **kwargs):
        def root_finding_function(lagrange_multiplier_input):
            self.lagrange_multiplier = lagrange_multiplier_input
            super(RHF, self).kernel(**kwargs)
            overlap = self.get_ovlp()
            sqrt_overlap = sqrtm(overlap)
            inverse_sqrt_overlap = np.linalg.inv(sqrt_overlap)

            dm = self.make_rdm1()

            eigenvectors = self.mo_coeff
            eigenvalues = self.mo_energy
            occupation = self.mo_occ

            bond_order = self.bond_order(overlap, dm)

            function_value = np.array(bond_order) - self.bond_order_constraint

            dmS = dm @ overlap
            aoslices = self.mol.aoslice_by_atom()
            Ls = list()
            for bond in self.bondatmlst:
                L = np.zeros_like(dm)
                a0, a1 = aoslices[bond[0], 2:]
                b0, b1 = aoslices[bond[1], 2:]
                tmp = overlap[:, b0:b1] @ dmS[b0:b1, a0:a1] / 2
                L[:, a0:a1] -= tmp
                L[a0:a1] -= tmp.T
                tmp = overlap[:, a0:a1] @ dmS[a0:a1, b0:b1] / 2
                L[:, b0:b1] -= tmp
                L[b0:b1] -= tmp.T
                Ls.append(L)

            density_matrix_derivatives = list()
            for L, bond, constraint in zip(Ls, bond_order, self.bond_order_constraint):
                outer_difference = np.subtract.outer(eigenvalues, eigenvalues)
                np.fill_diagonal(outer_difference, 1)
                eigenvalues_outer_inverse_difference = 1.0 / outer_difference
                np.fill_diagonal(eigenvalues_outer_inverse_difference, 0)
                transform_matrix = \
                    (bond - constraint) * (eigenvectors.T @ overlap @ eigenvectors) * eigenvalues_outer_inverse_difference

                transform_matrix_in_ao_basis = inverse_sqrt_overlap @ transform_matrix @ sqrt_overlap

                eigenvectors_derivative = np.transpose(transform_matrix_in_ao_basis @ eigenvectors.T)

                occupied_eigenvectors = eigenvectors[:, occupation > 0]
                occupied_eigenvectors_derivative = eigenvectors_derivative[:, occupation > 0]
                density_matrix_derivative = \
                    np.dot(occupied_eigenvectors * occupation[occupation > 0],
                           occupied_eigenvectors_derivative.conj().T) \
                    + np.dot(occupied_eigenvectors_derivative * occupation[occupation > 0],
                             occupied_eigenvectors.conj().T)

                density_matrix_derivatives.append(density_matrix_derivative)

            jacobians = list()
            for L in Ls:
                jacobian = list()
                for D in density_matrix_derivatives:
                    jacobian.append(np.vdot(D,L))
                jacobians.append(np.array(jacobian))

            print(jacobians)

            return function_value

        print(root(root_finding_function, self.lagrange_multiplier))

if __name__ == "__main__":
    from pyscf import gto, scf
    
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = '''
    O        0.000000    0.000000    0.117790
    H        0.000000    0.755453   -0.471161
    H        0.000000   -0.755453   -0.471161'''
    mol.basis = 'ccpvdz'
    mol.build()
    
    # from pyscf.constrained.scf.hf import CRHF
    mf = scf.RHF(mol)
    mf.kernel()
    dm0 = mf.make_rdm1()

    bond_atom_list = [[0,1]]
    required_bond_order = [-3e-1]
    mf = CRHF(mol, bond_atom_list, required_bond_order)
    mf.max_cycle = 100
    mf.kernel(dm0=dm0)
    
    s1e = mf.get_ovlp()

    assert(np.abs(np.max(np.array(mf.bond_order(s1e, mf.make_rdm1())) - np.array(required_bond_order))) < 1e-10)

