#!/usr/bin/env python
#
# Author: Chenghan Li <lch004218@gmail.com>
#

'''
Dual Basis Restricted Kohn Sham DFT
'''

from pyscf.scf import dualbase_hf
from pyscf.dft import rks

class DualBaseRKS(rks.RKS, dualbase_hf.DualBaseRHF):
   '''
   I want this class kernel calls DualBaseRHF.kernel
   but inside DualBaseRHF.kernel, I want get_veff calls the RKS one
   '''

   def __init__(self, mol, mol2, xc='LDA,VWN'):
      rks.RKS.__init__(self, mol, xc)

      if not mol2._built:
         sys.stderr.write('Warning: %s must be initialized before calling SCF.\n'
                           'Initialize %s in %s\n' % (mol2, mol2, self))
         mol2.build()
      self.mol2 = mol2


   kernel = dualbase_hf.DualBaseRHF.kernel

   def nuc_grad_method(self):
       from pyscf.grad import dualbase_rks_rough
       #return dualbase_rks_rough.GradientsHF(self)
       #return dualbase_rks_rough.GradientsNoU(self)
       return dualbase_rks_rough.GradientsFP(self)

class DualBaseRKSCG(rks.RKS, dualbase_hf.DualBaseRHF):
   '''
   I want this class kernel calls DualBaseRHF.kernel
   but inside DualBaseRHF.kernel, I want get_veff calls the RKS one
   '''

   def __init__(self, mol, mol2, gridlev1=2, gridlev2=None, xc='LDA,VWN'):
      rks.RKS.__init__(self, mol, xc)
      if gridlev2 is None:
        from pyscf import __config__
        gridlev2 = getattr(__config__, 'dft_rks_RKS_grids_level', 
                self.grids.level)
      self.grids.level = gridlev1
      self.gridlev1 = gridlev1
      self.gridlev2 = gridlev2

      if not mol2._built:
         sys.stderr.write('Warning: %s must be initialized before calling SCF.\n'
                           'Initialize %s in %s\n' % (mol2, mol2, self))
         mol2.build()
      self.mol2 = mol2


   kernel = dualbase_hf.DualBaseRHF.kernel

   def nuc_grad_method(self):
       from pyscf.grad import dualbase_rks_rough
       #return dualbase_rks_rough.GradientsHF(self)
       #return dualbase_rks_rough.GradientsNoU(self)
       return dualbase_rks_rough.GradientsFP(self)
