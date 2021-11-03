from pyscf.grad import dualbase_rhf_rough as dbrhfr_grad
from pyscf.grad import rks as rks_grad
from pyscf.lib import logger
import numpy

class GradientsHF(rks_grad.Gradients):


    __init__ = dbrhfr_grad.GradientsHF.__init__
    kernel = dbrhfr_grad.GradientsHF.kernel
    as_scanner = dbrhfr_grad.GradientsHF.as_scanner

