from pyscf import lib, gto
from pyscf.grad import rhf as rhf_grad
from pyscf.lib import logger
import numpy

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
