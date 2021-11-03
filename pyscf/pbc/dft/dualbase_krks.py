#!/usr/bin/env python
#
# Author: Chenghan Li <lch004218@gmail.com>
#

'''
Dual Basis Restricted Kohn Sham DFT for periodic systems with multiple k points
'''

from pyscf.pbc.scf import dualbase_hf
from pyscf.pbc.dft import krks
from pyscf import __config__

import numpy as np

class DualBaseKRKS(krks.KRKS, dualbase_hf.DualBaseRHF):
   '''
   I want this class kernel calls DualBaseRHF.kernel
   but inside DualBaseRHF.kernel, I want get_veff calls the RKS one
   '''

   def __init__(self, cell, cell2, kpts=np.zeros((1,3)), xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
      krks.KRKS.__init__(self, cell, kpts=kpts, xc=xc, exxdiv=exxdiv)

      if not cell2._built:
         sys.stderr.write('Warning: %s must be initialized before calling SCF.\n'
                           'Initialize %s in %s\n' % (cell2, cell2, self))
         cell2.build()
      self.cell2 = cell2

      self.mol = self.cell
      self.mol2 = self.cell2

      self._keys = self._keys.union(['cell2', 'mol2'])

   kernel = dualbase_hf.DualBaseRHF.kernel
