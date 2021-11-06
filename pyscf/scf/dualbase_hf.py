#!/usr/bin/env python
#
# Author: Chenghan Li <lch004218@gmail.com>
#

'''
Dual Basis Hartree-Fock
'''

import numpy

import h5py
from pyscf.scf import hf
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import addons

def kernel(mf, conv_tol=1e-10, conv_tol_grad=None,
           dump_chk=True, dm0=None, callback=None, conv_check=True, **kwargs):
    '''
    Examples:

    >>> from pyscf import gto, scf
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1', basis='cc-pvdz')
    >>> mol2 = gto.M(atom='H 0 0 0; H 0 0 1.1', basis='cc-pvtz')
    >>> mf = scf.dualbase_hf.DualBaseRHF(mol, mol2)
    >>> conv, e, mo_e, mo, mo_occ = mf.kernel()
    >>> print('conv = %s, E(HF) = %.12f' % (conv, e))
    '''

    # SCF using smaller basis in mol1
    conv, e, mf.mo_energy_small, mf.mo_coeff_small, mf.mo_occ_small = \
       hf.kernel(
           mf, conv_tol, conv_tol_grad, dump_chk, dm0, callback, conv_check, **kwargs)
    logger.info(mf, 'SCF on small basis E= %.15g', e)
    mf.converged = conv

    # project dm_small into dm_proj using larger basis in mol2
    dm_small = mf.make_rdm1(mf.mo_coeff_small, mf.mo_occ_small)
    dm_proj = addons.project_dm_nr2nr(mf.mol, dm_small, mf.mol2)

    # reset things for mol2
    mf._reset(mf.mol2)

    # construct fock from dm_proj
    h1e = mf.get_hcore(mf.mol2)
    vhf = mf.get_veff(mf.mol2, dm_proj)
    fock = mf.get_fock(h1e=h1e, vhf=vhf)
    mf.fock_proj = fock

    # diagonalization once
    s1e = mf.get_ovlp(mf.mol2)
    mf.mo_energy_large, mf.mo_coeff_large = mf.eig(fock, s1e)
    mf.mo_occ_large = mf.get_occ(mf.mo_energy_large, mf.mo_coeff_large)

    # these aliases can be useful for other routines
    mf.mo_energy = mf.mo_energy_large
    mf.mo_coeff = mf.mo_coeff_large
    mf.mo_occ = mf.mo_occ_large

    # update energy
    dm_large = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    new_vhf = mf.get_veff(mf.mol2, dm_large)
    e_tot = mf.energy_tot(dm_large, h1e, new_vhf)
    logger.info(mf, 'One SCF on larger basis E= %.15g', e_tot)
    # @@@@ ad hoc and ugly solution for mulitple k points
    if hasattr(mf, 'kpts'):
        mf.e_tot = e + \
              numpy.einsum('kij,kji->', dm_large-dm_proj, fock) / len(mf.kpts)
    else:
        mf.e_tot = e + numpy.einsum('ij,ji->', dm_large-dm_proj, fock)
    logger.info(mf, 'Corrected Energy on larger basis E= %.15g', mf.e_tot)

    # SCF to converge
    if 'scf2converge' in kwargs and kwargs.get('scf2converge') == True:
        mf.converged, mf.e_tot, mf.mo_energy, mf.mo_coeff, mf.mo_occ = \
              hf.kernel(mf, conv_tol, conv_tol_grad,
                        dump_chk, dm_large, callback, conv_check, **kwargs)

    # reset again for future call of this kernel
    mf._reset(mf.mol)

    return mf.converged, mf.e_tot, mf.mo_energy, mf.mo_coeff, mf.mo_occ


def as_scanner(mf):
    ''' copy paste from scf.hf.as_scanner
    '''
    if isinstance(mf, lib.SinglePointScanner):
        return mf

    logger.info(mf, 'Create scanner for %s', mf.__class__)

    class SCF_Scanner(mf.__class__, lib.SinglePointScanner):
        def __init__(self, mf_obj):
            self.__dict__.update(mf_obj.__dict__)

        def __call__(self, mol_or_geom, mol_or_geom2=None, **kwargs):
            if isinstance(mol_or_geom, gto.Mole):
                mol = mol_or_geom
            else:
                mol = self.mol.set_geom_(mol_or_geom, inplace=False)
            if mol_or_geom2 is not None:
                if isinstance(mol_or_geom2, gto.Mole):
                    mol2 = mol_or_geom2
                else:
                    mol2 = self.mol2.set_geom_(mol_or_geom2, inplace=False)
            else: # no mol2 given, assuming just updating coords
                mol2 = self.mol2.set_geom_(mol.atom_coords(), unit='Bohr', inplace=False)

            # Cleanup intermediates associated to the pervious mol object
            self.reset(mol)
            self.mol2 = mol2

            if 'dm0' in kwargs:
                dm0 = kwargs.pop('dm0')
            elif self.mo_coeff is None:
                dm0 = None
            elif self.chkfile and h5py.is_hdf5(self.chkfile):
                dm0 = self.from_chk(self.chkfile)
            else:
                dm0 = self.make_rdm1()
                # dm0 form last calculation cannot be used in the current
                # calculation if a completely different system is given.
                # Obviously, the systems are very different if the number of
                # basis functions are different.
                # TODO: A robust check should include more comparison on
                # various attributes between current `mol` and the `mol` in
                # last calculation.
                if dm0.shape[-1] != mol.nao:
                    # TODO:
                    #from pyscf.scf import addons
                    # if numpy.any(last_mol.atom_charges() != mol.atom_charges()):
                    #    dm0 = None
                    # elif non-relativistic:
                    #    addons.project_dm_nr2nr(last_mol, dm0, last_mol)
                    # else:
                    #    addons.project_dm_r2r(last_mol, dm0, last_mol)
                    dm0 = None
            self.mo_coeff = None  # To avoid last mo_coeff being used by SOSCF
            e_tot = self.kernel(dm0=dm0, **kwargs)
            return e_tot

    return SCF_Scanner(mf)

class DualBaseRHF(hf.RHF):

    def __init__(self, mol, mol2):
        '''
        I expect mol and mol2 only differ by basis set, but I do not check!
        '''
        super(hf.RHF, self).__init__(mol)

        if not mol2._built:
            sys.stderr.write('Warning: %s must be initialized before calling SCF.\n'
                              'Initialize %s in %s\n' % (mol2, mol2, self))
            mol2.build()

        self.mol2 = mol2

        self.mo_energy_small = None
        self.mo_coeff_small = None
        self.mo_occ_small = None
        self.mo_energy_large = None
        self.mo_coeff_large = None
        self.mo_occ_large = None

        self.fock_proj = None

        self._keys = self._keys.union(
            ['mol2', \
             'mo_energy_small', 'mo_coeff_small', 'mo_occ_small', \
             'mo_energy_large', 'mo_coeff_large', 'mo_occ_large',
             'fock_proj'] )

    def _reset(self, mol):
        #      self.reset(self.mol2)
        #      if hasattr(self, "mol"):
        #         self.mol = self.mol2
        #      if hasattr(self, "cell"):
        #         self.cell = self.cell2
        self._eri = None                # seems needed by pbc.scf.rhf ?
        if hasattr(self, "with_df"):
            self.with_df.__init__(mol)   # seems needed by pbc.scf.rhf ?
            if hasattr(self.with_df, "task_list"):
                self.with_df.task_list = None   # seems needed by FFTDF2?
        if hasattr(self, "grids"):
            self.grids.__init__(mol)     # in case DFT.reset doesn't do this
        if hasattr(self, "nlcgrids"):
            self.nlcgrids.__init__(mol)  # in case DFT.reset doesn't do this

    def kernel(self, dm0=None, **kwargs):
        ''' copy paste from scf.hf.scf()
        '''
        cput0 = (logger.process_clock(), logger.perf_counter())

        self.dump_flags()
        self.build(self.mol)

        if self.max_cycle > 0 or self.mo_coeff is None:
            self.converged, self.e_tot, \
                    self.mo_energy, self.mo_coeff, self.mo_occ = \
                    kernel(self, self.conv_tol, self.conv_tol_grad,
                           dm0=dm0, callback=self.callback,
                           conv_check=self.conv_check, **kwargs)
        else:
            # Avoid to update SCF orbitals in the non-SCF initialization
            # (issue #495).  But run regular SCF for initial guess if SCF was
            # not initialized.
            self.e_tot = kernel(self, self.conv_tol, self.conv_tol_grad,
                                dm0=dm0, callback=self.callback,
                                conv_check=self.conv_check, **kwargs)[1]

        logger.timer(self, 'SCF', *cput0)
        self._finalize()
        return self.e_tot

    def nuc_grad_method(self):
        from pyscf.grad import dualbase_rhf_rough
        #return dualbase_rhf_rough.GradientsHF(self)
        return dualbase_rhf_rough.GradientsNoU(self)

    as_scanner = as_scanner
