from pyscf import lib
from pyscf.dft import numint
from pyscf.grad import dualbase_rhf_rough as dbrhfr_grad
from pyscf.grad import rks as rks_grad
from pyscf.lib import logger
from pyscf.scf import addons
import numpy
import time


class GradientsHF(rks_grad.Gradients):


    __init__ = dbrhfr_grad.GradientsHF.__init__
    kernel = dbrhfr_grad.GradientsHF.kernel
    as_scanner = dbrhfr_grad.GradientsHF.as_scanner

class GradientsOneSCF(rks_grad.Gradients):

    def __init__(self, method):
        rks_grad.Gradients.__init__(self, method)
        self.mol2 = method.mol2

    def kernel(self, 
        mo_energy_large=None, mo_coeff_large=None, mo_occ_large=None,
        atmlst=None):
        self.base._reset(self.mol2)        # in case self.base is using mol

        if mo_energy_large is None: mo_energy_large = self.base.mo_energy_large
        if mo_coeff_large is None: mo_coeff_large = self.base.mo_coeff_large
        if mo_occ_large is None: mo_occ_large = self.base.mo_occ_large

        self.mol, self.mol2 = self.mol2, self.mol
        self.de = rks_grad.Gradients.kernel(self, 
                mo_energy=mo_energy_large, mo_coeff=mo_coeff_large,
                mo_occ=mo_occ_large, atmlst=atmlst)
        self.mol, self.mol2 = self.mol2, self.mol

        return self.de

def grad_elec_mn(mf_grad, m, n, mol=None, atmlst=None):
    '''
    Electronic part of Dual Base RHF/RKS gradients

    Args:
        mf_grad : grad.rhf.Gradients or grad.rks.Gradients object
    '''
    mf = mf_grad.base
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)

    # density-independent terms
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    assert m <= 1 and n <= 1
    mo_occ = mf.mo_occ_large

    # compute projected quantities (0-th order quantities)
    mo_coeff_0 = numpy.zeros(mf.mo_coeff_large.shape)
    mo_coeff_0[...,:mf.mol.nao] += \
        addons.project_mo_nr2nr(mf.mol, mf.mo_coeff_small, mf.mol2)
    dm_0 = mf.make_rdm1(mo_coeff_0, mo_occ)
    mo_energy_0 = numpy.einsum('ij,jk,ki->i', 
            mo_coeff_0.conj().T, mf.fock_proj, mo_coeff_0)
    mo_coeffs = [mo_coeff_0]
    mo_energys = [mo_energy_0]
    dms = [dm_0]

    # compute 1st order quantities
    mo_coeffs.append(mf.mo_coeff_large)
    mo_energys.append(mf.mo_energy_large)
    dms.append(mf.make_rdm1(mo_coeffs[-1], mo_occ))

    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Computing Gradients of NR-HF Coulomb repulsion')
    # build fock using n-th order of dm
    vxc, vhf_m, vhf_n = mf_grad.get_veff(mol, dms[m], dms[n])
    log.timer('gradients of 2e part', *t0)

    # I consider this approx the grad of tr(P_m F(P_n))
    dme = mf_grad.make_rdm1e(mo_energys[n], mo_coeffs[m], mo_occ)

    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de = numpy.zeros((len(atmlst),3))
    # @@@@@
#    mf_grad.h1ao_de = numpy.zeros((mf_grad.mol.natm, 3))
#    mf_grad.vhf_de = numpy.zeros((mf_grad.mol.natm, 3))
#    mf_grad.s1_de = numpy.zeros((mf_grad.mol.natm, 3))
    # @@@@@
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices [ia,2:]
        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum('xij,ij->x', h1ao, dms[m])
# nabla was applied on bra in vhf, *2 for the contributions of nabla|ket>
        de[k] += numpy.einsum('xij,ij->x', vhf_n[:,p0:p1], dms[m][p0:p1]) 
        de[k] += numpy.einsum('xij,ij->x', vhf_m[:,p0:p1], dms[n][p0:p1])
# grad of Exc = \int exc(rho_n) dr; non-zero total grad if use vxc(dm_n) \dot dm_m
        de[k] += numpy.einsum('xij,ij->x', vxc[:,p0:p1], dms[n][p0:p1]) * 2
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], dme[p0:p1]) * 2

        de[k] += mf_grad.extra_force(ia, locals())
        # @@@@@
#        mf_grad.h1ao_de[k] +=  numpy.einsum('xij,ij->x', h1ao, dms[m])
#        mf_grad.vhf_de[k] +=   numpy.einsum('xij,ij->x', vhf[:,p0:p1], dms[m][p0:p1]) * 2
#        mf_grad.s1_de[k] +=    numpy.einsum('xij,ij->x', s1[:,p0:p1], dme[p0:p1]) * 2
        # @@@@@                

    if log.verbose >= logger.DEBUG:
        log.debug('gradients of electronic part')
        _write(log, mol, de, atmlst)
    return de

class GradientsMN(rks_grad.Gradients):

    def __init__(self, method):
        rks_grad.Gradients.__init__(self, method)
        self.mol2 = method.mol2

    grad_elec_mn = grad_elec_mn

    def get_veff(self, mol, dm_m, dm_n):
        mf = self.base
        ni = mf._numint
        if self.grids is not None:
            grids = self.grids
        else:
            grids = mf.grids
        if grids.coords is None:
            grids.build(with_non0tab=True)
        if mf.nlc != '':
            raise NotImplementedError
        #enabling range-separated hybrids
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)

        exc, vxc = rks_grad.get_vxc(ni, mol, grids, mf.xc, dm_n)

        if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
            vhf_m = self.get_j(mol, dm_m)
            vhf_n = self.get_j(mol, dm_n)
        else:
            vj, vk = self.get_jk(mol, dm_m)
            vk *= hyb
            if abs(omega) > 1e-10:  # For range separated Coulomb operator
                with mol.with_range_coulomb(omega):
                    vk += self.get_k(mol, dm_m) * (alpha - hyb)
            vhf_m = vj - vk * .5
            vj, vk = self.get_jk(mol, dm_n)
            vk *= hyb
            if abs(omega) > 1e-10:  # For range separated Coulomb operator
                with mol.with_range_coulomb(omega):
                    vk += self.get_k(mol, dm_n) * (alpha - hyb)
            vhf_n = vj - vk * .5
        return vxc, vhf_m, vhf_n

    def kernel(self, m, n, atmlst=None):
        cput0 = (logger.process_clock(), logger.perf_counter())
        self.base._reset(self.mol2)
        
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        de = self.grad_elec_mn(m, n, mol=self.mol2, atmlst=atmlst)
        self.de = de + self.grad_nuc(atmlst=atmlst)
        if self.mol.symmetry:
            self.de = self.symmetrize(self.de, atmlst)
        logger.timer(self, 'SCF gradients', *cput0)
        self._finalize()
        return self.de

class GradientsNoU(rks_grad.Gradients):


    def __init__(self, method):
        rks_grad.Gradients.__init__(self, method)
        self.mol2 = method.mol2

    kernel = dbrhfr_grad.GradientsNoU.kernel
    as_scanner = dbrhfr_grad.GradientsNoU.as_scanner

    def add_grad_elec(self, \
        mo_energy_proj,  mo_coeff_proj,  
        mo_energy_large, mo_coeff_large, mo_occ_large, 
        dm_proj, dm_diff, atmlst):

        log = logger.Logger(self.stdout, self.verbose)
        hcore_deriv = self.hcore_generator(self.mol2)
        s1 = self.get_ovlp(self.mol2)

        #### get_veff ####
        ''' skeleton derivative of 'two-electron'-part of energy
        '''
        t0 = (logger.process_clock(), logger.perf_counter())

        #@@@@@@
        chhli_print_time = False
        if chhli_print_time: ttt = time.time()
        #@@@@@@
        mf = self.base
        ni = mf._numint

        if self.grids is not None:
            grids = self.grids
        else:
            grids = mf.grids
        if grids.coords is None:
            grids.build(with_non0tab=True)

        if mf.nlc != '':
            raise NotImplementedError
        #enabling range-separated hybrids
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=self.mol2.spin)

        mem_now = lib.current_memory()[0]
        max_memory = max(2000, self.max_memory*.9-mem_now)
        #@@@@@@
        if chhli_print_time: print(time.time() - ttt); ttt = time.time()
        #@@@@@@

        ### get_vxc ###
        ''' \partial tr(F^XC \Delta P) / \partial P_\mu_\nv
            this will have three parts
            (1) U F^XC will become the dme * S after absorbed into  U F
            (2) skeleton F^XC derivative
            (3) Fock-like Exc second derivative
            (4) Exc second derivative times U; assumed to be zero
        '''
#        ## skeleton derivatives of F^XC
#        if self.grid_response:
#            exc, vxc = rks_grad.get_vxc_full_response(\
#                    ni,self.mol2, grids, mf.xc, dm_proj,
#                    max_memory=max_memory,
#                    verbose=self.verbose)
#            logger.debug1(self, 'sum(grids response) %s', exc.sum(axis=0))
#        else:
#            # this one takes 0.28 s
#            exc, vxc = rks_grad.get_vxc(\
#                    ni, self.mol2, grids, mf.xc, dm_proj,
#                    max_memory=max_memory, 
#                    verbose=self.verbose)
        #@@@@@@
        if chhli_print_time: print(time.time() - ttt); ttt = time.time()
        #@@@@@@

        ## skeleton derivatives of F^XC and Exc second derivative
        relativity = 0
        spin = 0
        if self.grid_response:
            raise NotImplementedError
            logger.warn(self, 'WARN: grid_response for dual base in only '
                                'evaluated for skeleton derivs with '
                                'projected density')
        else:
            '''
            only bra derivs computed here; for bra+ket, simply times 2
            '''
            xc_code = mf.xc
            xctype = ni._xc_type(xc_code)
            make_rho, nset, nao = \
                    ni._gen_rho_evaluator(self.mol2, [dm_proj,dm_diff], hermi=1)
            ao_loc = self.mol2.ao_loc_nr()
            vxc  = numpy.zeros((3,nao,nao))
            vmat = numpy.zeros((3,nao,nao))
            if xctype == 'LDA':
                '''
                1. vxc will be contracted with dm_diff:
                \int fr (\nabla m)n dr
                2. vmat will be contracted with dm_proj:
                \sum_mn \Delta P_mn \int frr mn (\nabla s)l dr
                = \int frr \rho_diff (\nabla s)l dr
                '''
                ao_deriv = 1
                xc_deriv = 2
                for ao, mask, weight, coords in \
                    ni.block_loop(self.mol2, grids, nao, ao_deriv, max_memory):

                    rho = make_rho(0, ao, mask, 'GGA')
                    rho_diff = make_rho(1, ao[0], mask, 'LDA')
                    vv, fxc = ni.eval_xc(xc_code, rho, spin, relativity, xc_deriv, 
                            verbose=self.verbose)[1:3]
                    fr  = vv[0]
                    frr = fxc[0]

                    # 1:
                    aow = numpy.einsum('gn,g->gn', ao[0], weight*fr)
                    rks_grad._d1_dot_(\
                            vxc, self.mol2, ao[1:4], aow, mask, ao_loc, True)

                    # 2:
                    aow = numpy.einsum('g,g,gm->gm', 
                                    rho_diff, weight*frr, ao[0])
                    rks_grad._d1_dot_(\
                            vmat, self.mol2, ao[1:4], aow, mask, ao_loc, True)
            elif xctype == 'GGA':
                ''' 
                1. vxc

                2. vmat:
                    Part 1
                \int [ f_rr \rho_diff + 
                       2f_rg\nabla\rho\dot\nabla\rho_diff ] (sl)^R dr
                    Part 2
                \int [ 2f_rg\rho_diff\nabla\rho + 
                       4f_gg(\nabla\rho\dot\nabla\rho_diff)\nabla\rho
                       2f_g\nabla\rho_diff ] (\nabla(sl))^R dr
                '''
                #@@@@@@
                if chhli_print_time: print(time.time() - ttt); ttt = time.time()
                #@@@@@@
                ao_deriv = 2
                xc_deriv = 2
                # block_loop takes 0.11 s ???
                for ao, mask, weight, coords in \
                    ni.block_loop(self.mol2, grids, nao, ao_deriv, max_memory):
                    #@@@@@@
                    if chhli_print_time: print(time.time() - ttt); ttt = time.time()
                    #@@@@@@
                    # make rho 0.04 s
                    rho = make_rho(0, ao, mask, 'GGA')
                    rho_diff = make_rho(1, ao, mask, 'GGA')
                    #@@@@@@
                    if chhli_print_time: print(time.time() - ttt); ttt = time.time()
                    #@@@@@@

                    # eval_xc takes only 0.02 s !!
                    vv, fxc = ni.eval_xc(xc_code, rho, spin, relativity, 
                            xc_deriv, verbose=self.verbose)[1:3]
                    fr  = vv[0]
                    fg  = vv[1]
                    frr = fxc[0]
                    frg = fxc[1]
                    fgg = fxc[2]

                    # 1. vxc:
                    wv = numint._rks_gga_wv0(rho, vv, weight)
                    rks_grad._gga_grad_sum_(vxc, self.mol2, ao, wv, mask, ao_loc)

                    #@@@@@@
                    if chhli_print_time: print(time.time() - ttt); ttt = time.time()
                    #@@@@@@

                    # 2. vmat:

                    # aow and daow takes 0.02 s ???
                    aow = numpy.einsum('gm,g->gm', ao[0], weight)
                    daow = numpy.einsum('xgm,g->xgm', ao[1:4], weight)

                    # part 1
                    # drho_dot_drho_diff takes 0.0007 s
                    drho_dot_drhod = numpy.sum(rho[1:4] * rho_diff[1:4], axis=0)
                    # part1 0.0003 s
                    part1 = frr * rho_diff[0] + 2 * frg * drho_dot_drhod
                    # d1_dot 0.03 s
                    rks_grad._d1_dot_(vmat, self.mol2, 
                            ao[1:4], part1[:,None] * aow, 
                            mask, ao_loc, True)
                    #@@@@@@
                    if chhli_print_time: print(time.time() - ttt); ttt = time.time()
                    #@@@@@@

                    # part 2
                    # part 2 takes 0.0006 s
                    part2 = frg * rho_diff[0] + 2 * fgg * drho_dot_drhod
                    part2 = part2 * rho[1:4]
                    part2 += fg * rho_diff[1:4]
                    part2 *= 2
                    #@@@@@@
                    if chhli_print_time: print(time.time() - ttt); ttt = time.time()
                    #@@@@@@
                    # d1_dot 0.03 s
                    # \int s^R * part2 \dot \nabla l dr
                    rks_grad._d1_dot_(\
                            vmat, self.mol2, ao[1:4], 
                            numpy.einsum('xg,xgl->gl', part2, daow),
                            mask, ao_loc, True)
                    #@@@@@@
                    if chhli_print_time: print(time.time() - ttt); ttt = time.time()
                    #@@@@@@
                    # vmat takes 0.03 * 3 = 0.1 s
                    # \int (\nabla s)^R \dot part2 l dr
                    vmat[0] += numint._dot_ao_ao(self.mol2, 
                            # part2 \dot (sxx, syx, szx)
                            numpy.einsum('xg,xgs->gs', part2, ao[[4,5,6]]),
                            aow, mask, (0, self.mol2.nbas), ao_loc)
                    vmat[1] += numint._dot_ao_ao(self.mol2, 
                            # part2 \dot (sxy, syy, szy)
                            numpy.einsum('xg,xgs->gs', part2, ao[[5,7,8]]),
                            aow, mask, (0, self.mol2.nbas), ao_loc)
                    vmat[2] += numint._dot_ao_ao(self.mol2, 
                            # part2 \dot (sxz, syz, szz)
                            numpy.einsum('xg,xgs->gs', part2, ao[[6,8,9]]),
                            aow, mask, (0, self.mol2.nbas), ao_loc)
                    #@@@@@@
                    if chhli_print_time: print(time.time() - ttt); ttt = time.time()
                    #@@@@@@
            else:
                raise NotImplementedError()


            vmat = -vmat
            vxc  = -vxc
        ### end of get_vxc ###

        # j k 0.56 s
        # j k parts
        if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
            v_proj, v_diff = self.get_j(self.mol2, [dm_proj, dm_diff])
        else:
            vj, vk = self.get_jk(self.mol2, [dm_proj, dm_diff])
            vk *= hyb
            if abs(omega) > 1e-10:  # For range separated Coulomb operator
                with self.mol2.with_range_coulomb(omega):
                    vk += self.get_k(self.mol2, [dm_proj, dm_diff]) * (alpha-hyb)
            v_proj = vj[0] - vk[0] * .5
            v_diff = vj[1] - vk[1] * .5

        log.timer('gradients of 2e part', *t0)
        #### end of get_veff ####

        #@@@@@@
        if chhli_print_time: print(time.time() - ttt); ttt = time.time()
        #@@@@@@
        # dme 0.0001 s
        dme_proj = self.make_rdm1e(mo_energy_proj, mo_coeff_proj, mo_occ_large)
        dme_large = self.make_rdm1e(mo_energy_large, mo_coeff_large, mo_occ_large)
        #@@@@@@
        if chhli_print_time: print(time.time() - ttt); ttt = time.time()
        #@@@@@@

        if atmlst is None:
            atmlst = range(self.mol2.natm)
        aoslices = self.mol2.aoslice_by_atom()
        # @@@@@
        self.h1ao_de = numpy.zeros((self.mol2.natm, 3))
        self.vhf_de = numpy.zeros((self.mol2.natm, 3))
        self.s1_de = numpy.zeros((self.mol2.natm, 3))
        # @@@@@
        for k, ia in enumerate(atmlst):
            #  this chunk 0.00086 s * Natoms
            p0, p1 = aoslices [ia,2:]
            h1ao = hcore_deriv(ia)
            self.de[k] += numpy.einsum('xij,ij->x', h1ao, dm_diff)
            # nabla was applied on bra in vhf, *2 for the contributions of nabla|ket>
            # and 1/2 because we do not consider J, K themselve response
            # and *2 because J K are double in tr(PF)
            self.de[k] += numpy.einsum('xij,ij->x', v_proj[:,p0:p1], dm_diff[p0:p1]) * 2
            self.de[k] += numpy.einsum('xij,ij->x', v_diff[:,p0:p1], dm_proj[p0:p1]) * 2

            #@@@@@@
            if chhli_print_time: print(time.time() - ttt); ttt = time.time()
            #@@@@@@
            # this chunk 4.6E-5 * Natoms
            # *2 for ket
            # skeleton:
            self.de[k] += numpy.einsum('xij,ij->x', vxc[:,p0:p1], dm_diff[p0:p1]) * 2
            # fxc:
            self.de[k] += numpy.einsum('xij,ij->x', vmat[:,p0:p1], dm_proj[p0:p1]) * 2
            # diag(UF):
            self.de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], dme_large[p0:p1]) * 2
            self.de[k] += numpy.einsum('xij,ij->x', s1[:,p0:p1], dme_proj[p0:p1]) * 2
            #@@@@@@
            if chhli_print_time: print(time.time() - ttt); ttt = time.time()
            #@@@@@@

class GradientsNoVirt(GradientsNoU):
    '''
    gradient that assumes no virt-occ U
    '''
    def kernel(self, \
            mo_energy_small=None, mo_coeff_small=None, mo_occ_small=None,
            mo_energy_large=None, mo_coeff_large=None, mo_occ_large=None,
            fock_proj=None, atmlst=None):

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

        # compute everything that does not need U
        GradientsNoU.kernel(self, mo_energy_small, mo_coeff_small, mo_occ_small,
                                  mo_energy_large, mo_coeff_large, mo_occ_large,
                                  fock_proj, atmlst)

        self.base._reset(self.mol)     
        dm_small = self.base.make_rdm1(mo_coeff_small, mo_occ_small)

        self.base._reset(self.mol2)     
        dm_large = self.base.make_rdm1(mo_coeff_large, mo_occ_large)
        s1 = self.get_ovlp(self.mol2)
        mo_coeff_proj = addons.project_mo_nr2nr( \
                self.mol, mo_coeff_small, self.mol2) # shape Nao_L x Nao_S
        dm_proj = addons.project_dm_nr2nr(self.mol, dm_small, self.mol2)

        # TODO handle hybrid functional
        jdiff = self.base.get_j(self.mol2, dm_proj-dm_large)
        # occ.-occ. part of U times Jdiff
        juoo = 2 * numpy.einsum('mk,nj,xmn,mn->xm', 
                mo_coeff_proj, mo_coeff_proj, s1, jdiff)

        if atmlst is None:
            atmlst = range(self.mol2.natm)
        aoslices = self.mol2.aoslice_by_atom()
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices [ia,2:]
            # *2 for bra
            self.de[k] += numpy.sum(juoo[:,p0:p1], axis=1) * 2

        return self.de

class GradientsFP(rks_grad.Gradients):
    '''
    perturb the force directly
    '''

    def __init__(self, method):
        rks_grad.Gradients.__init__(self, method)
        self.mol2 = method.mol2

    kernel = dbrhfr_grad.GradientsNoU.kernel
    as_scanner = dbrhfr_grad.GradientsNoU.as_scanner

    def add_grad_elec(self, \
        mo_energy_proj,  mo_coeff_proj,  
        mo_energy_large, mo_coeff_large, mo_occ_large, 
        dm_proj, dm_diff, atmlst):

        full_s_diff = True
        full_vxc_diff = True
        full_j_diff = False

        log = logger.Logger(self.stdout, self.verbose)
        hcore_deriv = self.hcore_generator(self.mol2)
        s1 = self.get_ovlp(self.mol2)

        #### get_veff ####
        ''' skeleton derivative of 'two-electron'-part of energy
        '''
        t0 = (logger.process_clock(), logger.perf_counter())

        #@@@@@@
        chhli_print_time = False
        if chhli_print_time: ttt = time.time()
        #@@@@@@
        mf = self.base
        ni = mf._numint

        if self.grids is not None:
            grids = self.grids
        else:
            grids = mf.grids
        if grids.coords is None:
            grids.build(with_non0tab=True)

        if mf.nlc != '':
            raise NotImplementedError
        #enabling range-separated hybrids
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=self.mol2.spin)

        mem_now = lib.current_memory()[0]
        max_memory = max(2000, self.max_memory*.9-mem_now)
        #@@@@@@
        if chhli_print_time: print(time.time() - ttt); ttt = time.time()
        #@@@@@@

        ### get_vxc ###
        ''' \partial tr(F^XC \Delta P) / \partial P_\mu_\nv
            this will have three parts
            (1) U F^XC will become the dme * S after absorbed into  U F
            (2) skeleton F^XC derivative
            (3) Fock-like Exc second derivative
            (4) Exc second derivative times U; assumed to be zero
        '''
#        ## skeleton derivatives of F^XC
#        if self.grid_response:
#            exc, vxc = rks_grad.get_vxc_full_response(\
#                    ni,self.mol2, grids, mf.xc, dm_proj,
#                    max_memory=max_memory,
#                    verbose=self.verbose)
#            logger.debug1(self, 'sum(grids response) %s', exc.sum(axis=0))
#        else:
#            # this one takes 0.28 s
#            exc, vxc = rks_grad.get_vxc(\
#                    ni, self.mol2, grids, mf.xc, dm_proj,
#                    max_memory=max_memory, 
#                    verbose=self.verbose)
        #@@@@@@
        if chhli_print_time: print(time.time() - ttt); ttt = time.time()
        #@@@@@@

        ## skeleton derivatives of F^XC and Exc second derivative
        if not full_vxc_diff:
            relativity = 0
            spin = 0
            if self.grid_response:
                raise NotImplementedError
                logger.warn(self, 'WARN: grid_response for dual base in only '
                                    'evaluated for skeleton derivs with '
                                    'projected density')
            else:
                '''
                only bra derivs computed here; for bra+ket, simply times 2
                '''
                xc_code = mf.xc
                xctype = ni._xc_type(xc_code)
                make_rho, nset, nao = \
                        ni._gen_rho_evaluator(self.mol2, [dm_proj,dm_diff], hermi=1)
                ao_loc = self.mol2.ao_loc_nr()
                vxc  = numpy.zeros((3,nao,nao))
                vmat = numpy.zeros((3,nao,nao))
                if xctype == 'LDA':
                    '''
                    1. vxc will be contracted with dm_diff:
                    \int fr (\nabla m)n dr
                    2. vmat will be contracted with dm_proj:
                    \sum_mn \Delta P_mn \int frr mn (\nabla s)l dr
                    = \int frr \rho_diff (\nabla s)l dr
                    '''
                    ao_deriv = 1
                    xc_deriv = 2
                    for ao, mask, weight, coords in \
                        ni.block_loop(self.mol2, grids, nao, ao_deriv, max_memory):
    
                        rho = make_rho(0, ao, mask, 'GGA')
                        rho_diff = make_rho(1, ao[0], mask, 'LDA')
                        vv, fxc = ni.eval_xc(xc_code, rho, spin, relativity, xc_deriv, 
                                verbose=self.verbose)[1:3]
                        fr  = vv[0]
                        frr = fxc[0]
    
                        # 1:
                        aow = numpy.einsum('gn,g->gn', ao[0], weight*fr)
                        rks_grad._d1_dot_(\
                                vxc, self.mol2, ao[1:4], aow, mask, ao_loc, True)
    
                        # 2:
                        aow = numpy.einsum('g,g,gm->gm', 
                                        rho_diff, weight*frr, ao[0])
                        rks_grad._d1_dot_(\
                                vmat, self.mol2, ao[1:4], aow, mask, ao_loc, True)
                elif xctype == 'GGA':
                    ''' 
                    1. vxc
    
                    2. vmat:
                        Part 1
                    \int [ f_rr \rho_diff + 
                           2f_rg\nabla\rho\dot\nabla\rho_diff ] (sl)^R dr
                        Part 2
                    \int [ 2f_rg\rho_diff\nabla\rho + 
                           4f_gg(\nabla\rho\dot\nabla\rho_diff)\nabla\rho
                           2f_g\nabla\rho_diff ] (\nabla(sl))^R dr
                    '''
                    #@@@@@@
                    if chhli_print_time: print(time.time() - ttt); ttt = time.time()
                    #@@@@@@
                    ao_deriv = 2
                    xc_deriv = 2
                    # block_loop takes 0.11 s ???
                    for ao, mask, weight, coords in \
                        ni.block_loop(self.mol2, grids, nao, ao_deriv, max_memory):
                        #@@@@@@
                        if chhli_print_time: print(time.time() - ttt); ttt = time.time()
                        #@@@@@@
                        # make rho 0.04 s
                        rho = make_rho(0, ao, mask, 'GGA')
                        rho_diff = make_rho(1, ao, mask, 'GGA')
                        #@@@@@@
                        if chhli_print_time: print(time.time() - ttt); ttt = time.time()
                        #@@@@@@
    
                        # eval_xc takes only 0.02 s !!
                        vv, fxc = ni.eval_xc(xc_code, rho, spin, relativity, 
                                xc_deriv, verbose=self.verbose)[1:3]
                        fr  = vv[0]
                        fg  = vv[1]
                        frr = fxc[0]
                        frg = fxc[1]
                        fgg = fxc[2]
    
                        # 1. vxc:
                        wv = numint._rks_gga_wv0(rho, vv, weight)
                        rks_grad._gga_grad_sum_(vxc, self.mol2, ao, wv, mask, ao_loc)
    
                        #@@@@@@
                        if chhli_print_time: print(time.time() - ttt); ttt = time.time()
                        #@@@@@@
    
                        # 2. vmat:
    
                        # aow and daow takes 0.02 s ???
                        aow = numpy.einsum('gm,g->gm', ao[0], weight)
                        daow = numpy.einsum('xgm,g->xgm', ao[1:4], weight)
    
                        # part 1
                        # drho_dot_drho_diff takes 0.0007 s
                        drho_dot_drhod = numpy.sum(rho[1:4] * rho_diff[1:4], axis=0)
                        # part1 0.0003 s
                        part1 = frr * rho_diff[0] + 2 * frg * drho_dot_drhod
                        # d1_dot 0.03 s
                        rks_grad._d1_dot_(vmat, self.mol2, 
                                ao[1:4], part1[:,None] * aow, 
                                mask, ao_loc, True)
                        #@@@@@@
                        if chhli_print_time: print(time.time() - ttt); ttt = time.time()
                        #@@@@@@
    
                        # part 2
                        # part 2 takes 0.0006 s
                        part2 = frg * rho_diff[0] + 2 * fgg * drho_dot_drhod
                        part2 = part2 * rho[1:4]
                        part2 += fg * rho_diff[1:4]
                        part2 *= 2
                        #@@@@@@
                        if chhli_print_time: print(time.time() - ttt); ttt = time.time()
                        #@@@@@@
                        # d1_dot 0.03 s
                        # \int s^R * part2 \dot \nabla l dr
                        rks_grad._d1_dot_(\
                                vmat, self.mol2, ao[1:4], 
                                numpy.einsum('xg,xgl->gl', part2, daow),
                                mask, ao_loc, True)
                        #@@@@@@
                        if chhli_print_time: print(time.time() - ttt); ttt = time.time()
                        #@@@@@@
                        # vmat takes 0.03 * 3 = 0.1 s
                        # \int (\nabla s)^R \dot part2 l dr
                        vmat[0] += numint._dot_ao_ao(self.mol2, 
                                # part2 \dot (sxx, syx, szx)
                                numpy.einsum('xg,xgs->gs', part2, ao[[4,5,6]]),
                                aow, mask, (0, self.mol2.nbas), ao_loc)
                        vmat[1] += numint._dot_ao_ao(self.mol2, 
                                # part2 \dot (sxy, syy, szy)
                                numpy.einsum('xg,xgs->gs', part2, ao[[5,7,8]]),
                                aow, mask, (0, self.mol2.nbas), ao_loc)
                        vmat[2] += numint._dot_ao_ao(self.mol2, 
                                # part2 \dot (sxz, syz, szz)
                                numpy.einsum('xg,xgs->gs', part2, ao[[6,8,9]]),
                                aow, mask, (0, self.mol2.nbas), ao_loc)
                        #@@@@@@
                        if chhli_print_time: print(time.time() - ttt); ttt = time.time()
                        #@@@@@@
                else:
                    raise NotImplementedError()
    
    
                vmat = -vmat
                vxc  = -vxc

        if full_vxc_diff:
            # vxc_proj is identical to vxc
            exc, vxc_proj = rks_grad.get_vxc(\
                    ni, self.mol2, grids, mf.xc, dm_proj,
                    max_memory=max_memory, 
                    verbose=self.verbose)
            exc, vxc_large = rks_grad.get_vxc(\
                    ni, self.mol2, grids, mf.xc, dm_diff+dm_proj,
                    max_memory=max_memory, 
                    verbose=self.verbose)
        ### end of get_vxc ###

        # j k 0.56 s
        # j k parts
        if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
            v_proj, v_diff = self.get_j(self.mol2, [dm_proj, dm_diff])
        else:
            vj, vk = self.get_jk(self.mol2, [dm_proj, dm_diff])
            vk *= hyb
            if abs(omega) > 1e-10:  # For range separated Coulomb operator
                with self.mol2.with_range_coulomb(omega):
                    vk += self.get_k(self.mol2, [dm_proj, dm_diff]) * (alpha-hyb)
            v_proj = vj[0] - vk[0] * .5
            v_diff = vj[1] - vk[1] * .5

        log.timer('gradients of 2e part', *t0)
        #### end of get_veff ####

        #@@@@@@
        if chhli_print_time: print(time.time() - ttt); ttt = time.time()
        #@@@@@@
        # dme 0.0001 s
        if full_s_diff:
            dme_proj = self.make_rdm1e(mo_energy_proj, mo_coeff_proj, mo_occ_large)
            dme_large = self.make_rdm1e(mo_energy_large, mo_coeff_large, mo_occ_large)
        else:
            dme_proj = self.make_rdm1e( \
                    mo_energy_large - mo_energy_proj, 
                    mo_coeff_proj, mo_occ_large)
            dme_diff = self.make_rdm1e(mo_energy_proj, mo_coeff_large, mo_occ_large)
            dme_diff = dme_diff - \
                       self.make_rdm1e(mo_energy_proj, mo_coeff_proj, mo_occ_large)
            if full_vxc_diff:
                veff_proj = self.base.get_veff(mol=self.mol2, dm=dm_proj) 
                veff_large = self.base.get_veff(mol=self.mol2, dm=dm_diff+dm_proj) 
                fock_diff = veff_large - veff_proj
                mo_energy_diff = numpy.einsum('ij,jk,ki->i',
                        mo_coeff_proj.conj().T,
                        fock_diff,
                        mo_coeff_proj)
                dme_diff2 = self.make_rdm1e(mo_energy_diff, mo_coeff_proj, mo_occ_large)
            else:
                raise Exception()
        #@@@@@@
        if chhli_print_time: print(time.time() - ttt); ttt = time.time()
        #@@@@@@

        if atmlst is None:
            atmlst = range(self.mol2.natm)
        aoslices = self.mol2.aoslice_by_atom()
        # @@@@@
        self.h1ao_de = numpy.zeros((self.mol2.natm, 3))
        self.vhf_de = numpy.zeros((self.mol2.natm, 3))
        self.s1_de = numpy.zeros((self.mol2.natm, 3))
        # @@@@@
        for k, ia in enumerate(atmlst):
            #  this chunk 0.00086 s * Natoms
            p0, p1 = aoslices [ia,2:]
            h1ao = hcore_deriv(ia)
            self.de[k] += numpy.einsum('xij,ij->x', h1ao, dm_diff)
            # and 1/2 because we do not consider J, K themselve response
            # and *2 because J K are double in tr(PF)
            self.de[k] += numpy.einsum('xij,ij->x', v_proj[:,p0:p1], dm_diff[p0:p1]) * 2
            self.de[k] += numpy.einsum('xij,ij->x', v_diff[:,p0:p1], dm_proj[p0:p1]) * 2
            # nabla was applied on bra in vhf, *2 for the contributions of nabla|ket>
            if full_j_diff:
                self.de[k] += numpy.einsum('xij,ij->x', v_diff[:,p0:p1], dm_diff[p0:p1]) * 4

            #@@@@@@
            if chhli_print_time: print(time.time() - ttt); ttt = time.time()
            #@@@@@@
            # this chunk 4.6E-5 * Natoms
            # *2 for ket
            # skeleton:
            if not full_vxc_diff:
                self.de[k] += numpy.einsum('xij,ij->x', vxc[:,p0:p1], dm_diff[p0:p1]) * 2
            else:
                self.de[k] -= numpy.einsum('xij,ij->x', vxc_proj[:,p0:p1], dm_proj[p0:p1]) * 2
                self.de[k] += numpy.einsum('xij,ij->x', vxc_large[:,p0:p1], (dm_diff+dm_proj)[p0:p1]) * 2
            # fxc:
            if not full_vxc_diff:
                self.de[k] += numpy.einsum('xij,ij->x', vmat[:,p0:p1], dm_proj[p0:p1]) * 2
            # diag(UF):
            if full_s_diff:
                self.de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], dme_large[p0:p1]) * 2
                self.de[k] += numpy.einsum('xij,ij->x', s1[:,p0:p1], dme_proj[p0:p1]) * 2
            else:
                self.de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], dme_proj[p0:p1]) * 2
                self.de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], dme_diff[p0:p1]) * 2
                # TODO (C_proj.T fock_diff_ao C_proj)_ii S_ii^R
                self.de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], dme_diff2[p0:p1]) * 2
            #@@@@@@
            if chhli_print_time: print(time.time() - ttt); ttt = time.time()
            #@@@@@@
