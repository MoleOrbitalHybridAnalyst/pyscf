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

      self._keys = self._keys.union(['mol2'])

   kernel = dualbase_hf.DualBaseRHF.kernel
