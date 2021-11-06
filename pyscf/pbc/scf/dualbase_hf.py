#!/usr/bin/env python
#
# Author: Chenghan Li <lch004218@gmail.com>
#

'''
Hartree-Fock for periodic systems at a single k-point with dual basis
'''

from pyscf.pbc.scf import hf
from pyscf.scf import dualbase_hf as mol_dbhf
from pyscf import __config__

class DualBaseRHF(hf.RHF, mol_dbhf.DualBaseRHF):

    def __init__(self, cell, cell2, kpt=[0,0,0],
                  exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        '''
        I expect mol and mol2 only differ by basis set, but I do not check!
        '''
        hf.RHF.__init__(self, cell, kpt=kpt, exxdiv=exxdiv)

        if not cell2._built:
            sys.stderr.write('Warning: cell2.build() is not called in input\n')
            cell2.build()

        self.cell2 = cell2

        self.mol = self.cell
        self.mol2 = self.cell2

        self._keys = self._keys.union(['cell2', 'mol2'])

    kernel = mol_dbhf.DualBaseRHF.kernel
