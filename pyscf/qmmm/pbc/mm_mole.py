import numpy
import numpy as np
from pyscf import gto
from pyscf import pbc
from pyscf import qmmm
from pyscf.gto.mole import is_au
from pyscf.data.elements import charge
from pyscf.lib import param, logger
from pyscf.pbc.gto.cell import _cut_mesh_for_ewald
from scipy.special import erf, erfc
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
    def __init__(self, atoms, a, rcut=20.0, grid_spacing=2.0, charges=None, zeta=None):
        pbc.gto.Cell.__init__(self)
        self.atom = self._atom = atoms
        self.unit = 'Bohr'
        self.charge_model = 'point'
        self.a = a
        self.rcut = rcut
        self.mesh = np.ceil(np.diag(self.lattice_vectors()) / grid_spacing).astype(int)

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
        self._built = True

    def get_zetas(self):
        if self.charge_model == 'gaussian':
            return self._env[self._atm[:,gto.PTR_ZETA]]
        else:
            return 1e16

    def get_ewald_params(self, precision=None, rcut=None):
        if rcut is None:
            ew_cut = self.rcut
        else:
            ew_cut = rcut
        if precision is None:
            precision = self.precision
        ew_eta = numpy.sqrt(max(numpy.log(4*numpy.pi*ew_cut**2/precision)/ew_cut**2, .1))
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

        b = self.reciprocal_vectors(norm_to=1)
        heights_inv = lib.norm(b, axis=1)
        rcut = self.rcut
        nimgs = np.ceil(rcut * heights_inv).astype(int)
        Lall = self.get_lattice_Ls(rcut=ew_cut, nimgs=nimgs)

        rLij = coords1[:,None,:] - coords2[None,:,:] + Lall[:,None,None,:]
        r = np.sqrt(lib.einsum('Lijx,Lijx->Lij', rLij, rLij))
        rLij = None
        r[r<1e-16] = 1e200
#        ewovrl = .5 * np.einsum('i,j,Lij->', chargs, chargs, erfc(ew_eta * r) / r)
        if charges2 is not None:
            ewovrl = -lib.einsum('j,Lij->i', charges2, erf(ew_eta * r) / r)
        else:
            ewovrl = -np.sum(erf(ew_eta * r) / r, axis=0)

        # last line of Eq. (F.5) in Martin
#        ewself  = -.5 * np.dot(chargs,chargs) * 2 * ew_eta / np.sqrt(np.pi)
#        if self.dimension == 3:
#            ewself += -.5 * np.sum(chargs)**2 * np.pi/(ew_eta**2 * self.vol)

        if charges2 is not None:
            ewself = 0
        else:
#            ewself = -.5 * np.eye(len(coords1)) * 2 * ew_eta / np.sqrt(np.pi)
            ewself = -np.eye(len(coords1)) * 2 * ew_eta / np.sqrt(np.pi)

        # g-space sum (using g grid) (Eq. (F.6) in Martin, but note errors as below)
        # Eq. (F.6) in Martin is off by a factor of 2, the
        # exponent is wrong (8->4) and the square is in the wrong place
        #
        # Formula should be
        #   1/2 * 4\pi / Omega \sum_I \sum_{G\neq 0} |ZS_I(G)|^2 \exp[-|G|^2/4\eta^2]
        # where
        #   ZS_I(G) = \sum_a Z_a exp (i G.R_a)

        Gv, Gvbase, weights = self.get_Gv_weights(mesh)
        absG2 = lib.einsum('gi,gi->g', Gv, Gv)
        absG2[absG2==0] = 1e200

        coulG = 4*np.pi / absG2
        coulG *= weights
#        ZSI = np.einsum("i,ij->j", chargs, self.get_SI(Gv))
#        ZexpG2 = ZSI * np.exp(-absG2/(4*ew_eta**2))
#        ewg = .5 * np.einsum('i,i,i', ZSI.conj(), ZexpG2, coulG).real
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

def create_mm_mol(atoms_or_coords, a, charges=None, radii=None, unit='Angstrom'):
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

    if not is_au(unit):
        a = a / param.BOHR

    return Cell(atoms, a, charges=charges, zeta=zeta)

create_mm_cell = create_mm_mol
