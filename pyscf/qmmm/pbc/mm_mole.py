import numpy
import numpy as np
from pyscf import gto
from pyscf import pbc
from pyscf import qmmm
from pyscf.gto.mole import is_au
from pyscf.data.elements import charge
from pyscf.lib import param, logger
from pyscf.pbc.gto.cell import _cut_mesh_for_ewald
from scipy.special import erf, erfc, lambertw
from pyscf import lib

class Cell(qmmm.mm_mole.Mole, pbc.gto.Cell):
    '''Cell class for MM particles.

    Args:
        atoms : geometry of MM particles (unit Bohr).

            | [[atom1, (x, y, z)],
            |  [atom2, (x, y, z)],
            |  ...
            |  [atomN, (x, y, z)]]

    Kwargs:
        charges : 1D array
            fractional charges of MM particles
        zeta : 1D array
            Gaussian charge distribution parameter.
            rho(r) = charge * Norm * exp(-zeta * r^2)

    '''
    def __init__(self, atoms, a, 
            rcut_ewald=20.0, rcut_dip=None,
            charges=None, zeta=None):
        pbc.gto.Cell.__init__(self)
        self.atom = self._atom = atoms
        self.unit = 'Bohr'
        self.charge_model = 'point'
        self.a = a
        self.rcut_ewald = rcut_ewald
        self.rcut_dip = rcut_dip

        # Initialize ._atm and ._env to save the coordinates and charges and
        # other info of MM particles
        natm = len(atoms)
        _atm = numpy.zeros((natm,6), dtype=numpy.int32)
        _atm[:,gto.CHARGE_OF] = [charge(a[0]) for a in atoms]
        coords = numpy.asarray([a[1] for a in atoms], dtype=numpy.double)
        if charges is None:
            _atm[:,gto.NUC_MOD_OF] = gto.NUC_POINT
            charges = _atm[:,gto.CHARGE_OF:gto.CHARGE_OF+1]
        else:
            _atm[:,gto.NUC_MOD_OF] = gto.NUC_FRAC_CHARGE
            charges = numpy.asarray(charges)[:,numpy.newaxis]

        self._env = numpy.append(numpy.zeros(gto.PTR_ENV_START),
                                 numpy.hstack((coords, charges)).ravel())
        _atm[:,gto.PTR_COORD] = gto.PTR_ENV_START + numpy.arange(natm) * 4
        _atm[:,gto.PTR_FRAC_CHARGE] = gto.PTR_ENV_START + numpy.arange(natm) * 4 + 3

        if zeta is not None:
            self.charge_model = 'gaussian'
            zeta = numpy.asarray(zeta, dtype=float).ravel()
            self._env = numpy.append(self._env, zeta)
            _atm[:,gto.PTR_ZETA] = gto.PTR_ENV_START + natm*4 + numpy.arange(natm)

        self._atm = _atm

        eta, _ = self.get_ewald_params()
        e = self.precision
        Q = np.sum(self.atom_charges()**2)
        L = self.vol**(1/3)
        kmax = np.sqrt(3)*eta/2/np.pi * np.sqrt(lambertw( 4*Q**(2/3)/3/np.pi**(2/3)/L**2/eta**(2/3) / e**(4/3) ).real)
        self.mesh = np.ceil(np.diag(self.lattice_vectors()) * kmax).astype(int) * 2 + 1

        self._built = True

    def get_lattice_Ls(self):
        Ts = lib.cartesian_prod((np.arange(-1, 2),
                                 np.arange(-1, 2),
                                 np.arange(-1, 2)))
        Lall = np.dot(Ts, self.lattice_vectors())
        return Lall

    def get_ewald_params(self, precision=None, rcut=None):
        if rcut is None:
            ew_cut = self.rcut_ewald
        else:
            ew_cut = rcut
        if precision is None:
            precision = self.precision
        e = precision
        Q = np.sum(self.atom_charges()**2)
        ew_eta = 1 / ew_cut * np.sqrt(lambertw(1/e*np.sqrt(Q/2/self.vol)).real)
        return ew_eta, ew_cut

    def get_ewald_pot(self, coords1, coords2=None, charges2=None):
        assert self.dimension == 3
        assert (coords2 is None and charges2 is None) or (coords2 is not None and charges2 is not None)

        if charges2 is not None:
            assert len(charges2) == len(coords2)
        else:
            coords2 = coords1

        ew_eta, ew_cut = self.get_ewald_params()
        mesh = self.mesh

        if charges2 is not None:
            Lall = self.get_lattice_Ls()
        else:
            Lall = np.zeros((1,3))

#        breakpoint()
        all_coords2 = lib.direct_sum('jx-Lx->Ljx', coords2, Lall).reshape(-1,3)
        if charges2 is not None:
            all_charges2 = np.hstack([charges2] * len(Lall))
        else:
            all_charges2 = None
        dist2 = lib.direct_sum('jx-x->jx', all_coords2, np.mean(coords1, axis=0))
        dist2 = lib.einsum('jx,jx->j', dist2, dist2)
        R = lib.direct_sum('ix-jx->ijx', coords1, all_coords2)
        r = np.sqrt(lib.einsum('ijx,ijx->ij', R, R))
        r[r<1e-16] = np.inf
        # TODO handle this cutoff more elegantly
#        r[r>ew_cut] = 1e60

        # TODO
        # [√] check both qm and mm real-space Coulomb energy correctness
        # [√] check mm_ewald_pot ew_cut independency
        # [√] check mm_ewald energy correctness
        # [?] check qm_ewald_pot ew_cut independency
        # [?] check qm_ewald_pot energy correctness

#        breakpoint()
        # substract the real-space Coulomb within rcut_dip
        mask = dist2 <= self.rcut_dip**2
        if all_charges2 is not None:
            Tij = 1 / r[:,mask]
            Rij = R[:,mask]
            Tija = -lib.einsum('ijx,ij->ijx', Rij, Tij**3)
            charges = all_charges2[mask]
            # TODO ewovrl0: pc-dip pc-quad 
            # TODO ewovrl1: dip-dip dip-quad
            # TODO ewovrl2: quad-pc quad-dip quad-quad
            # fuv = dQi/druv ew0i
            # fuv = dDia/druv ew1ia
            # qm pc - mm pc
            ewovrl0 = -lib.einsum('ij,j->i', Tij, charges)
            # qm dip - mm pc
            ewovrl1 = -lib.einsum('j,ija->ia', charges, Tija)
#            return np.zeros_like(ewovrl0), np.zeros_like(ewovrl1)
#            return ewovrl0, ewovrl1
        else:
            assert mask.all()              # all qm atoms should be within rcut_dip
            assert r.shape[0] == r.shape[1]   # real-space should not see qm images
            Tij = 1 / r
            Tija = -lib.einsum('ijx,ij->ijx', R, Tij**3)
            Tijab  = 3 * lib.einsum('ija,ijb->ijab', R, R) 
            Tijab  = lib.einsum('ijab,ij->ijab', Tijab, Tij**5)
            Tijab -= lib.einsum('ij,ab->ijab', Tij**3, np.eye(3))
            # ew0 = -d^2 E / dQi dQj
            # ew1 = -d^2 E / dQi dDja
            # ew2 = -d^2 E / dDia dDjb
            # TODO beyond dip
            ewovrl0 = -Tij
            ewovrl1 =  Tija
            ewovrl2 =  Tijab
#            return ewovrl0, ewovrl1, ewovrl2

        # ewald real-space sum
        if all_charges2 is not None:
            ekR = np.exp(-ew_eta**2 * r**2)
            # Tij = \hat{1/r} = f0 / r = erfc(r) / r
            Tij = erfc(ew_eta * r) / r
            # Tija = -Rija \hat{1/r^3} = -Rija / r^2 ( \hat{1/r} + 2 eta/sqrt(pi) exp(-eta^2 r^2) )
            invr3 = -(Tij + 2 * ew_eta / np.sqrt(np.pi) * ekR) / r**2
            Tija = lib.einsum('ijx,ij->ijx', R, invr3)

            ewovrl0 += lib.einsum('ij,j->i', Tij, all_charges2)
            ewovrl1 += lib.einsum('j,ija->ia', all_charges2, Tija)
        else:
            ekR = np.exp(-ew_eta**2 * r**2)
            # Tij = \hat{1/r} = f0 / r = erfc(r) / r
            Tij = erfc(ew_eta * r) / r
            # Tija = -Rija \hat{1/r^3} = -Rija / r^2 ( \hat{1/r} + 2 eta/sqrt(pi) exp(-eta^2 r^2) )
            invr3 = (Tij + 2*ew_eta/np.sqrt(np.pi) * ekR) / r**2
            Tija = -lib.einsum('ijx,ij->ijx', R, invr3)
            # Tijab = (3 RijaRijb - Rij^2 delta_ab) \hat{1/r^5}
            Tijab  = 3 * lib.einsum('ija,ijb,ij->ijab', R, R, 1/r**2)
            Tijab -= lib.einsum('ij,ab->ijab', np.ones_like(r), np.eye(3))
            invr5 = invr3 + 4/3*ew_eta**3/np.sqrt(np.pi) * ekR # NOTE this is invr5 * r**2
            Tijab = lib.einsum('ijab,ij->ijab', Tijab, invr5)
            # NOTE the below is present in Eq 8 but missing in Eq 12
            Tijab += 4/3*ew_eta**3/np.sqrt(np.pi)*lib.einsum('ij,ab->ijab', ekR, np.eye(3))

            ewovrl0 += Tij
            ewovrl1 -= Tija
            ewovrl2 -= Tijab

        if all_charges2 is not None:
            ewself0 = 0
            ewself1 = 0
        else:
            ewself1 = 0
            # -d^2 Eself / dQi dQj
            ewself0 = -np.eye(len(coords1)) * 2 * ew_eta / np.sqrt(np.pi)
            # -d^2 Eself / dDia dDjb
            ewself2 = -lib.einsum('ij,ab->ijab', np.eye(len(coords1)), np.eye(3)) \
                    * 4 * ew_eta**3 / 3 / np.sqrt(np.pi)

        # g-space sum (using g grid)

        Gv, Gvbase, weights = self.get_Gv_weights(mesh)
        absG2 = lib.einsum('gx,gx->g', Gv, Gv)
        absG2[absG2==0] = 1e200

        coulG = 4*np.pi / absG2
        coulG *= weights
        # NOTE Gpref is actually Gpref*2
        Gpref = np.exp(-absG2/(4*ew_eta**2)) * coulG

        GvR2 = lib.einsum('gx,ix->ig', Gv, coords2)
        cosGvR2 = np.cos(GvR2)
        sinGvR2 = np.sin(GvR2)

        if charges2 is not None:
            GvR1 = lib.einsum('gx,ix->ig', Gv, coords1)
            cosGvR1 = np.cos(GvR1)
            sinGvR1 = np.sin(GvR1)
            zcosGvR2 = lib.einsum("i,ig->g", charges2, cosGvR2)
            zsinGvR2 = lib.einsum("i,ig->g", charges2, sinGvR2)
            # TODO ewg0: pc-dip pc-quad
            # qm pc - mm pc
            ewg0  = lib.einsum('ig,g,g->i', cosGvR1, zcosGvR2, Gpref)
            ewg0 += lib.einsum('ig,g,g->i', sinGvR1, zsinGvR2, Gpref)
            # TODO ewg1: dip-dip dip-quad
            # qm dip - mm pc
            ewg1  = lib.einsum('gx,ig,g,g->ix', Gv, cosGvR1, zsinGvR2, Gpref)
            ewg1 -= lib.einsum('gx,ig,g,g->ix', Gv, sinGvR1, zcosGvR2, Gpref)
            # TODO ewg2: quad-pc quad-dip quad-quad
        else:
            # qm pc - qm pc
            ewg0  = lib.einsum('ig,jg,g->ij', cosGvR2, cosGvR2, Gpref)
            ewg0 += lib.einsum('ig,jg,g->ij', sinGvR2, sinGvR2, Gpref)
            # qm pc - qm dip
            ewg1  = lib.einsum('gx,ig,jg,g->ijx', Gv, sinGvR2, cosGvR2, Gpref)
            ewg1 -= lib.einsum('gx,ig,jg,g->ijx', Gv, cosGvR2, sinGvR2, Gpref)
            # qm dip - qm dip
            ewg2  = lib.einsum('gx,gy,ig,jg,g->ijxy', Gv, Gv, cosGvR2, cosGvR2, Gpref)
            ewg2 += lib.einsum('gx,gy,ig,jg,g->ijxy', Gv, Gv, sinGvR2, sinGvR2, Gpref)

        if charges2 is not None:
            return ewovrl0 + ewg0, ewovrl1 + ewg1
        else:
#            # @@@@@@@@@@
#            np.save("ewovrl0", ewovrl0)
#            np.save("ewself0", ewself0)
#            np.save("ewg0", ewg0)
#            np.save("ewovrl1", ewovrl1)
#            np.save("ewg1", ewg1)
#            np.save("ewovrl2", ewovrl2)
#            np.save("ewself2", ewself2)
#            np.save("ewg2", ewg2)
#            return ewovrl0 + ewself0 + ewg0, ewovrl1 + ewself1 + ewg1, np.zeros_like(ewovrl2 + ewself2 + ewg2)
#            # @@@@@@@@@@
            return ewovrl0 + ewself0 + ewg0, ewovrl1 + ewself1 + ewg1, ewovrl2 + ewself2 + ewg2

def create_mm_mol(atoms_or_coords, a, charges=None, radii=None, 
        rcut_ewald=None, rcut_dip=None, unit='Angstrom'):
    '''Create an MM object based on the given coordinates and charges of MM
    particles.

    Args:
        atoms_or_coords : array-like
            Cartesian coordinates of MM atoms, in the form of a 2D array:
            [(x1, y1, z1), (x2, y2, z2), ...]
        a : (3,3) ndarray
            Lattice primitive vectors. Each row represents a lattice vector
            Reciprocal lattice vectors are given by  b1,b2,b3 = 2 pi inv(a).T

    Kwargs:
        charges : 1D array
            The charges of MM atoms.
        radii : 1D array
            The Gaussian charge distribuction radii of MM atoms.
        unit : string
            The unit of the input. Default is 'Angstrom'.
    '''
    if isinstance(atoms_or_coords, numpy.ndarray):
        # atoms_or_coords == np.array([(xx, xx, xx)])
        # Patch ghost atoms
        atoms = [(0, c) for c in atoms_or_coords]
    elif (isinstance(atoms_or_coords, (list, tuple)) and
          atoms_or_coords and
          isinstance(atoms_or_coords[0][1], (int, float))):
        # atoms_or_coords == [(xx, xx, xx)]
        # Patch ghost atoms
        atoms = [(0, c) for c in atoms_or_coords]
    else:
        atoms = atoms_or_coords
    atoms = gto.format_atom(atoms, unit=unit)

    if radii is None:
        zeta = None
    else:
        radii = numpy.asarray(radii, dtype=float).ravel()
        if not is_au(unit):
            radii = radii / param.BOHR
        zeta = 1 / radii**2

    kwargs = {'charges': charges, 'zeta': zeta}

    if not is_au(unit):
        a = a / param.BOHR
        if rcut_ewald is not None:
            rcut_ewald = rcut_ewald / param.BOHR
        if rcut_dip is not None:
            rcut_dip = rcut_dip / param.BOHR

    if rcut_ewald is not None:
        kwargs['rcut_ewald'] = rcut_ewald
    if rcut_dip is not None:
        kwargs['rcut_dip'] = rcut_dip

    return Cell(atoms, a, **kwargs)

create_mm_cell = create_mm_mol
