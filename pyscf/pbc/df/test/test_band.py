# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy
from pyscf.pbc import gto, scf, df

cell = gto.M(atom='H 1 2 1; H 1 1 1', a=numpy.eye(3)*4, verbose=0, mesh=[11]*3)
numpy.random.seed(1)
kband = numpy.random.random((2,3))

def finger(a):
    a = numpy.asarray(a)
    return numpy.cos(numpy.arange(a.size)).dot(a.ravel())

class KnowValues(unittest.TestCase):
    def test_fft_band(self):
        mf = scf.RHF(cell)
        mf.kernel()
        self.assertAlmostEqual(finger(mf.get_bands(kband[0])[0]), -0.037522993818908168, 8)

    def test_aft_band(self):
        mf = scf.RHF(cell)
        mf.with_df = df.AFTDF(cell)
        mf.kernel()
        self.assertAlmostEqual(finger(mf.get_bands(kband[0])[0]), -0.093770552709347754, 8)

    def test_df_band(self):
        mf = scf.RHF(cell)
        mf.with_df = df.DF(cell).set(auxbasis='weigend')
        mf.with_df.kpts_band = kband[0]
        mf.kernel()
        self.assertAlmostEqual(finger(mf.get_bands(kband[0])[0]), -0.093621142380270361, 8)

    def test_mdf_band(self):
        mf = scf.RHF(cell)
        mf.with_df = df.MDF(cell).set(auxbasis='weigend')
        mf.with_df.kpts_band = kband[0]
        mf.kernel()
        self.assertAlmostEqual(finger(mf.get_bands(kband[0])[0]), -0.093705179423150875, 6)

    def test_fft_bands(self):
        mf = scf.KRHF(cell)
        mf.kpts = cell.make_kpts([2]*3)
        mf.kernel()
        self.assertAlmostEqual(finger(mf.get_bands(kband[0])[0]), -0.26539150034518982, 8)
        self.assertAlmostEqual(finger(mf.get_bands(kband)[0]), -0.6562304384317044, 8)

    def test_aft_bands(self):
        mf = scf.KRHF(cell)
        mf.with_df = df.AFTDF(cell)
        mf.kpts = cell.make_kpts([2]*3)
        mf.kernel()
        self.assertAlmostEqual(finger(mf.get_bands(kband[0])[0]), -0.32214958122356513, 8)
        self.assertAlmostEqual(finger(mf.get_bands(kband)[0]), -0.64207984898611348, 8)

    def test_df_bands(self):
        mf = scf.KRHF(cell)
        mf.with_df = df.DF(cell).set(auxbasis='weigend')
        mf.with_df.kpts_band = kband
        mf.kpts = cell.make_kpts([2]*3)
        mf.kernel()
        self.assertAlmostEqual(finger(mf.get_bands(kband[0])[0]), -0.32200476157045782, 8)
        self.assertAlmostEqual(finger(mf.get_bands(kband)[0]), -0.64216552168395347, 8)

    def test_mdf_bands(self):
        mf = scf.KRHF(cell)
        mf.with_df = df.MDF(cell).set(auxbasis='weigend')
        mf.with_df.kpts_band = kband
        mf.kpts = cell.make_kpts([2]*3)
        mf.kernel()
        self.assertAlmostEqual(finger(mf.get_bands(kband[0])[0]), -0.32205949082575414, 7)
        self.assertAlmostEqual(finger(mf.get_bands(kband)[0]), -0.6420939191777898, 8)


if __name__ == '__main__':
    print("Full Tests for bands")
    unittest.main()

