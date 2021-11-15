from pyscf.grad import dualbase_rhf_rough as dbrhfr_grad
from pyscf.grad import rks as rks_grad
from pyscf.lib import logger
import numpy

class GradientsHF(rks_grad.Gradients):


    __init__ = dbrhfr_grad.GradientsHF.__init__
    kernel = dbrhfr_grad.GradientsHF.kernel
    as_scanner = dbrhfr_grad.GradientsHF.as_scanner

class GradientsNoU(rks_grad.Gradients):


    def __init__(self, method):
        rks_grad.Gradients.__init__(self, method)
        self.mol2 = method.mol2

#    kernel = dbrhfr_grad.GradientsNoU.kernel
    as_scanner = dbrhfr_grad.GradientsNoU.as_scanner

    def kernel(self, \
            mo_energy_small=None, mo_coeff_small=None, mo_occ_small=None,
            mo_energy_large=None, mo_coeff_large=None, mo_occ_large=None,
            fock_proj=None, atmlst=None):
        cput0 = (logger.process_clock(), logger.perf_counter())
        self.base._reset(self.mol)      # TODO in case self.base is still using mol2

        if mo_energy_small is None: mo_energy_small = self.base.mo_energy_small
        if mo_coeff_small is None: mo_coeff_small = self.base.mo_coeff_small
        if mo_occ_small is None: mo_occ_small = self.base.mo_occ_small
        if mo_energy_large is None: mo_energy_large = self.base.mo_energy_large
        if mo_coeff_large is None: mo_coeff_large = self.base.mo_coeff_large
        if mo_occ_large is None: mo_occ_large = self.base.mo_occ_large
        if fock_proj is None: fock_proj = self.base.fock_proj
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        # compute the standard gradient in small basis
        # I want to call rhf_grad.Gradients.kernel here
#        self.de = super().kernel(
        self.de = super(rhf_grad.Gradients, self).kernel(
                mo_energy=mo_energy_small, \
                mo_coeff=mo_coeff_small, \
                mo_occ=mo_occ_small, atmlst=atmlst)
        self.de_small = self.de.copy()

        self.base._reset(self.mol2)     # TODO just like in scf.dualbse_hf, we need this 

        # compute projected mo_coeff C and dm
        mo_coeff_proj = numpy.zeros(mo_coeff_large.shape)
        mo_coeff_proj[...,:self.mol.nao] += \
            addons.project_mo_nr2nr(
                    self.mol, mo_coeff_small, self.mol2) # shape Nao_L x Nao_S
        # replicated calc. of dm_proj; should be the same as self.base.dm_proj
        dm_proj = self.base.make_rdm1(mo_coeff_proj, mo_occ_large)

        # compute F(P) in C representation: C^T F(P)_AO C
        # and take the diagonal term as the "mo_energy"
        # Note we only need the occupied MO terms
        mo_energy_proj = numpy.einsum('ij,jk,ki->i', 
                                      mo_coeff_proj.conj().T,
                                      fock_proj,
                                      mo_coeff_proj)

        # will use P_diff to compute P_diff h^R 
        dm_diff = self.base.make_rdm1(mo_coeff_large, mo_occ_large) - dm_proj

        ## adapted from pyscf.grad.rhf.grad_elec ##
        log = logger.Logger(self.stdout, self.verbose)
        hcore_deriv = self.hcore_generator(self.mol2)
        s1 = self.get_ovlp(self.mol2)

        t0 = (logger.process_clock(), logger.perf_counter())
        #vhf = self.get_veff(self.mol2, dm_proj)  # get j, k derivs of fock_proj
        v_proj, v_diff = self.get_veff(self.mol2, [dm_proj, dm_diff])  # get j, k derivs of fock_proj
        log.timer('gradients of 2e part', *t0)

        dme_proj = self.make_rdm1e(mo_energy_proj, mo_coeff_proj, mo_occ_large)
        dme_large = self.make_rdm1e(mo_energy_large, mo_coeff_large, mo_occ_large)

        if atmlst is None:
            atmlst = range(self.mol2.natm)
        aoslices = self.mol2.aoslice_by_atom()
        # @@@@@
        self.h1ao_de = numpy.zeros((self.mol2.natm, 3))
        self.vhf_de = numpy.zeros((self.mol2.natm, 3))
        self.s1_de = numpy.zeros((self.mol2.natm, 3))
        # @@@@@
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices [ia,2:]
            h1ao = hcore_deriv(ia)
            self.de[k] += numpy.einsum('xij,ij->x', h1ao, dm_diff)
            # nabla was applied on bra in vhf, *2 for the contributions of nabla|ket>
            # and 1/2 because we do not consider J, K themselve response
            # and *2 because J K are double in tr(PF)
#            self.de[k] += numpy.einsum('xij,ij->x', vhf[:,p0:p1], dm_diff[p0:p1]) * 2
#            self.de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], dme_large[p0:p1]) * 2
#            self.de[k] += numpy.einsum('xij,ij->x', s1[:,p0:p1], dme_proj[p0:p1]) * 2
            self.de[k] += numpy.einsum('xij,ij->x', v_proj[:,p0:p1], dm_diff[p0:p1]) * 2
            self.de[k] += numpy.einsum('xij,ij->x', v_diff[:,p0:p1], dm_proj[p0:p1]) * 2
            self.de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], dme_large[p0:p1]) * 2
            self.de[k] += numpy.einsum('xij,ij->x', s1[:,p0:p1], dme_proj[p0:p1]) * 2
            # @@@@@@@@@@@@@
            self.h1ao_de[k] += numpy.einsum('xij,ij->x', h1ao, dm_diff)
            self.vhf_de[k] += numpy.einsum('xij,ij->x', v_proj[:,p0:p1], dm_diff[p0:p1]) 
            self.vhf_de[k] += numpy.einsum('xij,ij->x', v_diff[:,p0:p1], dm_proj[p0:p1]) 
            self.s1_de[k] += (numpy.einsum('xij,ij->x', s1[:,p0:p1], dme_proj[p0:p1]) - numpy.einsum('xij,ij->x', s1[:,p0:p1], dme_large[p0:p1]))
            # @@@@@@@@@@@@@
#            if k == 0:
#                print(numpy.einsum('xij,ij->x', h1ao, dm_diff), numpy.einsum('xij,ij->x', vhf[:,p0:p1], dm_diff[p0:p1]), numpy.einsum('xij,ij->x', -s1[:,p0:p1], dme_large[p0:p1]), numpy.einsum('xij,ij->x', s1[:,p0:p1], dme_proj[p0:p1]))
    
            # we have alreay done this when comput. grad in small basis SCF
            # de[k] += self.extra_force(ia, locals()) 

        ## end of grad_elec ##
        # @@@@@@@@@@@
        self.dm_proj = dm_proj
        self.dm_large = dm_proj + dm_diff
        self.dme_proj = dme_proj
        self.dme_large = dme_large
        self.mol, self.mol2 = self.mol2, self.mol
        save = self.de.copy()
#        self.de_large = super().kernel(
        self.de_large = super(rhf_grad.Gradients, self).kernel(
                mo_energy=mo_energy_large, \
                mo_coeff=mo_coeff_large, \
                mo_occ=mo_occ_large, atmlst=atmlst)
        self.de = save
        self.mol, self.mol2 = self.mol2, self.mol
        # @@@@@@@@@@@

        if self.mol.symmetry:
            self.de = self.symmetrize(self.de, atmlst)
        logger.timer(self, 'SCF gradients', *cput0)
        self._finalize()
        return self.de
