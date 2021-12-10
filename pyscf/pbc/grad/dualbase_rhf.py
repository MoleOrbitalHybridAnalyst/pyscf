from pyscf.pbc.grad import rhf as rhf_grad

class GradientsOneSCF(rhf_grad.Gradients):

    def __init__(self, method):
        rhf_grad.Gradients.__init__(self, method)
        self.mol2 = method.mol2
        self.cell2 = method.cell2

    def kernel(self,
            mo_energy_large=None, mo_coeff_large=None, mo_occ_large=None,
            atmlst=None, cache_mol2=False):

        if not self.base.cache_mol2:
            # mf.cache_mol2 is False
            # means mf._reset(mol) was called in mf.kernel()
            # so we need to reset to mol2 here
            self.base._reset(self.mol2)

        if mo_energy_large is None: mo_energy_large = self.base.mo_energy_large
        if mo_coeff_large is None: mo_coeff_large = self.base.mo_coeff_large
        if mo_occ_large is None: mo_occ_large = self.base.mo_occ_large

        self.mol, self.mol2 = self.mol2, self.mol
        self.de = rhf_grad.Gradients.kernel(self,
                mo_energy=mo_energy_large, mo_coeff=mo_coeff_large,
                mo_occ=mo_occ_large, atmlst=atmlst)
        self.mol, self.mol2 = self.mol2, self.mol

        self.base._reset(self.mol)

        return self.de
