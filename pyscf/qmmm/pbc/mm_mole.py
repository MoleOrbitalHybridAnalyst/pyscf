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
    def __init__(self, atoms, a, rcut=20.0, charges=None, zeta=None):
        pbc.gto.Cell.__init__(self)
        self.atom = self._atom = atoms
        self.unit = 'Bohr'
        self.charge_model = 'point'
        self.a = a
        self.rcut = rcut

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

    def get_ewald_params(self, precision=None, rcut=None):
        if rcut is None:
            ew_cut = self.rcut
        else:
            ew_cut = rcut
        if precision is None:
            precision = self.precision
        e = precision
        Q = np.sum(self.atom_charges()**2)
        ew_eta = 1 / ew_cut * np.sqrt(lambertw(1/e*np.sqrt(Q/2/self.vol)).real)
        return ew_eta, ew_cut

    def get_ewald_pot(self, coords1, coords2=None, charges2=None, Lall0=False):
        assert self.dimension == 3
        assert (coords2 is None and charges2 is None) or (coords2 is not None and charges2 is not None)

        if charges2 is not None:
            assert len(charges2) == len(coords2)
        else:
            coords2 = coords1

        ew_eta, ew_cut = self.get_ewald_params()
        mesh = self.mesh

        b = self.reciprocal_vectors(norm_to=1)
        heights_inv = lib.norm(b, axis=1)
        rcut = self.rcut
        nimgs = np.ceil(rcut * heights_inv).astype(int)
        Lall = self.get_lattice_Ls(rcut=ew_cut, nimgs=nimgs)
        if Lall0:
            Lall = np.zeros((1,3))

        rLij = coords1[:,None,:] - coords2[None,:,:] + Lall[:,None,None,:]
        r = np.sqrt(lib.einsum('Lijx,Lijx->Lij', rLij, rLij))
        rLij = None
        r[r<1e-16] = 1e200
        if charges2 is not None:
            ewovrl = -lib.einsum('j,Lij->i', charges2, erf(ew_eta * r) / r)
        else:
            ewovrl = -np.sum(erf(ew_eta * r) / r, axis=0)

        if charges2 is not None:
            ewself = 0
        else:
            ewself = -np.eye(len(coords1)) * 2 * ew_eta / np.sqrt(np.pi)

        # g-space sum (using g grid)

        Gv, Gvbase, weights = self.get_Gv_weights(mesh)
        absG2 = lib.einsum('gi,gi->g', Gv, Gv)
        absG2[absG2==0] = 1e200

        coulG = 4*np.pi / absG2
        coulG *= weights
        SI = self.get_SI(Gv, coords2)
        if charges2 is not None:
            SI1 = self.get_SI(Gv, coords1)
            ZSI = lib.einsum("i,ig->g", charges2, SI)
            ZexpG2 = ZSI * np.exp(-absG2/(4*ew_eta**2))
            ewg = lib.einsum('ig,g,g->i', SI1.conj(), ZexpG2, coulG).real
        else:
            expG2 = lib.einsum("ig,g->ig", SI, np.exp(-absG2/(4*ew_eta**2)))
            ewg = lib.einsum('ig,jg,g->ij', SI.conj(), expG2, coulG).real

        return ewovrl + ewself + ewg

def create_mm_mol(atoms_or_coords, a, charges=None, radii=None, rcut=None, unit='Angstrom'):
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
        if rcut is not None:
            rcut = rcut / param.BOHR

    if rcut is not None:
        kwargs['rcut'] = rcut

    return Cell(atoms, a, **kwargs)

create_mm_cell = create_mm_mol
