from pyscf import lib
from pyscf.dft import numint
from pyscf.grad import dualbase_rhf_rough as dbrhfr_grad
from pyscf.grad import rks as rks_grad
from pyscf.lib import logger
import numpy
import time


class GradientsHF(rks_grad.Gradients):


    __init__ = dbrhfr_grad.GradientsHF.__init__
    kernel = dbrhfr_grad.GradientsHF.kernel
    as_scanner = dbrhfr_grad.GradientsHF.as_scanner

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
        '''
        ## skeleton derivatives of F^XC
        if self.grid_response:
            exc, vxc = rks_grad.get_vxc_full_response(\
                    ni,self.mol2, grids, mf.xc, dm_proj,
                    max_memory=max_memory,
                    verbose=self.verbose)
            logger.debug1(self, 'sum(grids response) %s', exc.sum(axis=0))
        else:
            # this one takes 0.28 s
            exc, vxc = rks_grad.get_vxc(\
                    ni, self.mol2, grids, mf.xc, dm_proj,
                    max_memory=max_memory, 
                    verbose=self.verbose)
        # @@@@@@@
        self.exc = exc
        self.vxc = vxc
        # @@@@@@@
        #@@@@@@
        if chhli_print_time: print(time.time() - ttt); ttt = time.time()
        #@@@@@@

        ## Exc second derivative
        relativity = 0
        spin = 0
        if self.grid_response:
            logger.warn(self, 'WARN: grid_response for dual base in only '
                                'evaluated for skeleton derivs with '
                                'projected density')
        if False:
            pass
#        if self.grid_response:
#            raise NotImplementedError()
        else:
            '''
            only bra derivs computed here; for bra+ket, simply times 2
            '''
            xc_code = mf.xc
            xctype = ni._xc_type(xc_code)
            make_rho, nset, nao = \
                    ni._gen_rho_evaluator(self.mol2, [dm_proj,dm_diff], hermi=1)
            ao_loc = self.mol2.ao_loc_nr()
            vmat = numpy.zeros((3,nao,nao))
            if xctype == 'LDA':
                '''
                \sum_mn \Delta P_mn \int frr mn (\nabla s)l dr
                = \int frr \rho_diff (\nabla s)l dr
                '''
                ao_deriv = 1
                xc_deriv = 2
                for ao, mask, weight, coords in \
                    ni.block_loop(self.mol2, grids, nao, ao_deriv, max_memory):

                    rho = make_rho(0, ao, mask, 'GGA')
                    rho_diff = make_rho(1, ao[0], mask, 'LDA')
                    fxc = ni.eval_xc(xc_code, rho, spin, relativity, xc_deriv, 
                            verbose=self.verbose)[2]
                    frr = fxc[0]
                    aow = numpy.einsum('g,g,gm->gm', 
                                    rho_diff, weight*frr, ao[0])
                    rks_grad._d1_dot_(\
                            vmat, self.mol2, ao[1:4], aow, mask, ao_loc, True)
            elif xctype == 'GGA':
                ''' Part 1
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
                    #@@@@@@
                    if chhli_print_time: print(time.time() - ttt); ttt = time.time()
                    #@@@@@@

                    # aow and daow takes 0.02 s ???
                    aow = ao[0] * weight[:,None]
                    daow = ao[1:4] * weight[:,None]

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

