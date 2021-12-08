from pyscf import lib, gto
from pyscf.scf import addons, _vhf
from pyscf.grad import rhf as rhf_grad
from pyscf.lib import logger
import numpy
import time

def as_scanner(mf_grad):
    raise NotImplementedError()
    ''' copy paste from grad.rhf.as_scanner
                    and scf.dualbase_hf.as_scanner
    '''
    if isinstance(mf_grad, lib.GradScanner):
        return mf_grad

    logger.info(mf_grad, 'Create scanner for %s', mf_grad.__class__)

    class SCF_GradScanner(mf_grad.__class__, lib.GradScanner):
        def __init__(self, g):
            lib.GradScanner.__init__(self, g)

        def __call__(self, mol_or_geom, mol_or_geom2=None, **kwargs):
            if isinstance(mol_or_geom, gto.Mole):
                mol = mol_or_geom
            else:
                raise Exception("TODO: check why geom is not working for RKS")
                mol = self.mol.set_geom_(mol_or_geom, inplace=False)
            if mol_or_geom2 is not None:
                if isinstance(mol_or_geom2, gto.Mole):
                    mol2 = mol_or_geom2
                else:
                    mol2 = self.mol2.set_geom_(mol_or_geom2, inplace=False)
            else: # no mol2 given, assuming just updating coords
                mol2 = self.mol2.set_geom_(mol.atom_coords(), unit='Bohr', inplace=False)

            mf_scanner = self.base
            e_tot = mf_scanner(mol)
            self.mol = mol
            self.mol2 = mol2

            # If second integration grids are created for RKS and UKS
            # gradients
            if getattr(self, 'grids', None):
                self.grids.reset(mol)

            de = self.kernel(**kwargs)
            return e_tot, de
    return SCF_GradScanner(mf_grad)

class GradientsHF(rhf_grad.Gradients):

    def __init__(self, method):
        rhf_grad.Gradients.__init__(self, method)
        self.mol2 = method.mol2

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        cput0 = (logger.process_clock(), logger.perf_counter())
        self.base._reset(self.mol2)     # TODO just like in scf.dualbse_hf, we need this 
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst
        if atmlst is None:
            atmlst = range(self.mol2.natm)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        # basically hcore_generator without get_hcore
        with_x2c = getattr(self.base, 'with_x2c', None)
        if with_x2c:
            Exception("I don't know what this is yet")
        else:
            with_ecp = self.mol2.has_ecp()
            if with_ecp:
                ecp_atoms = set(self.mol2._ecpbas[:,gto.ATOM_OF])
            else:
                ecp_atoms = ()
            aoslices = self.mol2.aoslice_by_atom()
            dm0 = self.base.make_rdm1(mo_coeff, mo_occ)
            de = numpy.zeros((len(atmlst),3))
            for k, ia in enumerate(atmlst):
                p0, p1 = aoslices [ia,2:]
                with self.mol2.with_rinv_at_nucleus(ia):
                    vrinv = self.mol2.intor('int1e_iprinv', comp=3) # <\nabla|1/r|>
                    vrinv *= -self.mol2.atom_charge(ia)
                    if with_ecp and ia in ecp_atoms:
                        vrinv += self.mol2.intor('ECPscalar_iprinv', comp=3)
                vrinv = vrinv + vrinv.transpose(0,2,1)
                de[k] += numpy.einsum('xij,ij->x', vrinv, dm0)

        self.de = de + self.grad_nuc(atmlst=atmlst)
        if self.mol2.symmetry:
            self.de = self.symmetrize(self.de, atmlst)
        logger.timer(self, 'SCF gradients', *cput0)
        self._finalize()
        return self.de

    as_scanner = as_scanner

class GradientsNoU(rhf_grad.Gradients):

    def __init__(self, method):
        rhf_grad.Gradients.__init__(self, method)
        self.mol2 = method.mol2


    def kernel(self, \
            mo_energy_small=None, mo_coeff_small=None, mo_occ_small=None,
            mo_energy_large=None, mo_coeff_large=None, mo_occ_large=None,
            fock_proj=None, atmlst=None, accurate_grid=True):
        cput0 = (logger.process_clock(), logger.perf_counter())
        self.base._reset(self.mol)      # TODO in case self.base is still using mol2
        #@@@@@@
        chhli_print_time = False
        if chhli_print_time: ttt = time.time()
        #@@@@@@

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
        #@@@@@@
        if chhli_print_time: print(time.time() - ttt); ttt = time.time()
        #@@@@@@

        # I want the small-basis grad to be accurate
        if accurate_grid and hasattr(self.base, 'gridlev1'):
            self.base.grids.level = self.base.gridlev2
        # gradient on small takes 0.06 s
        # compute the standard gradient in small basis
        # I want to call rhf_grad.Gradients.kernel here
#        self.de = super().kernel(
        self.de = super(rhf_grad.Gradients, self).kernel(
                mo_energy=mo_energy_small, \
                mo_coeff=mo_coeff_small, \
                mo_occ=mo_occ_small, atmlst=atmlst)
        self.de_small = self.de.copy()

        self.base._reset(self.mol2)     # TODO just like in scf.dualbse_hf, we need this 
        #@@@@@@
        if chhli_print_time: print(time.time() - ttt); ttt = time.time()
        #@@@@@@

        # takes 0.001 s
        # compute projected mo_coeff C and dm
        mo_coeff_proj = numpy.zeros(mo_coeff_large.shape)
        mo_coeff_proj[...,:self.mol.nao] += \
            addons.project_mo_nr2nr(
                    self.mol, mo_coeff_small, self.mol2) # shape Nao_L x Nao_S
        # replicated calc. of dm_proj; should be the same as self.base.dm_proj
        dm_proj = self.base.make_rdm1(mo_coeff_proj, mo_occ_large)
        #@@@@@@
        if chhli_print_time: print(time.time() - ttt); ttt = time.time()
        #@@@@@@

        # this block takes 0.0005 s
        # compute F(P) in C representation: C^T F(P)_AO C
        # and take the diagonal term as the "mo_energy"
        # Note we only need the occupied MO terms
        mo_energy_proj = numpy.einsum('ij,jk,ki->i', 
                                      mo_coeff_proj.conj().T,
                                      fock_proj,
                                      mo_coeff_proj)

        # will use P_diff to compute P_diff h^R 
        dm_diff = self.base.make_rdm1(mo_coeff_large, mo_occ_large) - dm_proj
        #@@@@@@
        if chhli_print_time: print(time.time() - ttt); ttt = time.time()
        #@@@@@@

        # this takes 1.21 s for GGA DFT
        # add elec part of grad for energy correction
        self.add_grad_elec(
            mo_energy_proj,  mo_coeff_proj,  
            mo_energy_large, mo_coeff_large, mo_occ_large, 
            dm_proj, dm_diff, atmlst)
        #@@@@@@
        if chhli_print_time: print(time.time() - ttt); ttt = time.time()
        #@@@@@@

        # @@@@@@@@@@@
#        self.dm_proj = dm_proj
#        self.dm_large = dm_proj + dm_diff
##        self.dme_proj = dme_proj
##        self.dme_large = dme_large
#        self.mol, self.mol2 = self.mol2, self.mol
#        save = self.de.copy()
##        self.de_large = super().kernel(
#        self.de_large = super(rhf_grad.Gradients, self).kernel(
#                mo_energy=mo_energy_large, \
#                mo_coeff=mo_coeff_large, \
#                mo_occ=mo_occ_large, atmlst=atmlst)
#        self.de = save
#        self.mol, self.mol2 = self.mol2, self.mol
        # @@@@@@@@@@@

        if self.mol.symmetry:
            self.de = self.symmetrize(self.de, atmlst)
        logger.timer(self, 'SCF gradients', *cput0)
        self._finalize()
        return self.de

    def add_grad_elec(self, \
            mo_energy_proj,  mo_coeff_proj,  
            mo_energy_large, mo_coeff_large, mo_occ_large, 
            dm_proj, dm_diff,
            atmlst=None):
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
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices [ia,2:]
            h1ao = hcore_deriv(ia)
            self.de[k] += numpy.einsum('xij,ij->x', h1ao, dm_diff)
            # nabla was applied on bra in vhf, *2 for the contributions of nabla|ket>
            # and 1/2 because we do not consider J, K themselve response
            # and *2 because J K are double in tr(PF)
            self.de[k] += numpy.einsum('xij,ij->x', v_proj[:,p0:p1], dm_diff[p0:p1]) * 2
            self.de[k] += numpy.einsum('xij,ij->x', v_diff[:,p0:p1], dm_proj[p0:p1]) * 2
            self.de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], dme_large[p0:p1]) * 2
            self.de[k] += numpy.einsum('xij,ij->x', s1[:,p0:p1], dme_proj[p0:p1]) * 2

class GradientsDiagF(GradientsNoU):
    pass
