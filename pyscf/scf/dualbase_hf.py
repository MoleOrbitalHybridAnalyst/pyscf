#!/usr/bin/env python
#
# Author: Chenghan Li <lch004218@gmail.com>
#

'''
Dual Basis Hartree-Fock
'''

import numpy 

from pyscf.scf import hf
from pyscf.lib import logger
from pyscf.scf import addons

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


   def kernel(self, conv_tol=1e-10, conv_tol_grad=None,
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
      conv, e, mo_e, mo, mo_occ = hf.kernel(self, conv_tol, conv_tol_grad, 
            dump_chk, dm0, callback, conv_check, **kwargs)
      logger.info(self, 'SCF on small basis E= %.15g', e)

      # project dm_small into dm_proj using larger basis in mol2
      dm_small = self.make_rdm1(mo, mo_occ)
      dm_proj = addons.project_dm_nr2nr(self.mol, dm_small, self.mol2)

      # reset things for mol2
#      self.reset(self.mol2)
      if hasattr(self, "mol"):
         self.mol = self.mol2
      if hasattr(self, "cell"):
         self.cell = self.cell2
      self._eri = None                      # seems needed by pbc.scf.rhf ?
      if hasattr(self, "with_df"):
         self.with_df.__init__(self.mol2)   # seems needed by pbc.scf.rhf ?
         if hasattr(self.with_df, "task_list"):
            self.with_df.task_list = None   # seems needed by FFTDF2?
      if hasattr(self, "grids"):
         self.grids.__init__(self.mol2)     # in case DFT.reset doesn't do this
      if hasattr(self, "nlcgrids"):
         self.nlcgrids.__init__(self.mol2)  # in case DFT.reset doesn't do this

      # construct fock from dm_proj
      h1e = self.get_hcore(self.mol2)
      vhf = self.get_veff(self.mol2, dm_proj)
      fock = self.get_fock(h1e=h1e, vhf=vhf)

      # diagonalization once
      s1e = self.get_ovlp(self.mol2)
      mo_energy, mo_coeff = self.eig(fock, s1e)
      mo_occ = self.get_occ(mo_energy, mo_coeff)

      # update energy
      dm_large = self.make_rdm1(mo_coeff, mo_occ)
#      logger.info(self, 'tr(\Delta PF)= %.15g', numpy.trace((dm_large-dm_proj)@fock))
      new_vhf = self.get_veff(self.mol2, dm_large)
      e_tot = self.energy_tot(dm_large, h1e, new_vhf)
      logger.info(self, 'One SCF on larger basis E= %.15g', e_tot)
      e_tot = e + numpy.einsum('ij,ji->', dm_large-dm_proj, fock)
      logger.info(self, 'Corrected Energy on larger basis E= %.15g', e_tot)

      # SCF to converge
      if 'scf2converge' in kwargs and kwargs.get('scf2converge') == True:
#         self.mol = self.mol2
         conv, e_tot, mo_energy, mo_coeff, mo_occ = \
               hf.kernel(self, conv_tol, conv_tol_grad, 
               dump_chk, dm_large, callback, conv_check, **kwargs)

      return conv, e_tot, mo_energy, mo_coeff, mo_occ
