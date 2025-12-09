import random
import math
import numpy as np
import sys

if __name__ == '__main__':
    sys.path.append('..')
    sys.path.append('.')

from pyzx.graph.scalar import DyadicNumber
from pyzx.simulate import MAGIC_GLOBAL, MAGIC_B60, MAGIC_B66, MAGIC_E6, MAGIC_O6, MAGIC_K6, MAGIC_PHI


def test_mul():
    d1 = DyadicNumber(1, 1, 0, 0, 1)
    d2 = DyadicNumber(1, 0, 1, 1, 0)
    d3 = d1 * d2
    assert np.isclose(d3.to_complex(), d1.to_complex() * d2.to_complex())

    d1 = DyadicNumber(0, 0, 1, 0, 1)
    d2 = DyadicNumber(0, 0, 1, 0, 1)
    d3 = d1 * d2
    assert np.isclose(d3.to_complex(), d1.to_complex() * d2.to_complex())


def test_mul_with_random_scalars():
    for _ in range(20):
        d1 = DyadicNumber(*[random.randint(-20, 20) for _ in range(5)])
        d2 = DyadicNumber(*[random.randint(-20, 20) for _ in range(5)])
        d3 = d1 * d2
        assert np.isclose(d3.to_complex(), d1.to_complex() * d2.to_complex())


def test_sqrt2():
    sqrt2 = DyadicNumber.sqrt2()
    assert np.isclose(sqrt2.to_complex(), np.sqrt(2))


def test_one():
    one = DyadicNumber.one()
    assert np.isclose(one.to_complex(), 1)


def test_conjugate():
    for _ in range(20):
        d1 = DyadicNumber(*[random.randint(-20, 20) for _ in range(5)])
        d1_conj = d1.conjugate()
        assert np.isclose(d1.to_complex(), d1_conj.to_complex().conjugate())


def test_magic_constants():
    sq2 = math.sqrt(2)
    MAGIC_GLOBAL2 = -(7 + 5 * sq2) / (2 + 2j)
    MAGIC_B602 = -16 + 12 * sq2
    MAGIC_B662 = 96 - 68 * sq2
    MAGIC_E62 = 10 - 7 * sq2
    MAGIC_O62 = -14 + 10 * sq2
    MAGIC_K62 = 7 - 5 * sq2
    MAGIC_PHI2 = 10 - 7 * sq2

    assert MAGIC_GLOBAL2 == MAGIC_GLOBAL.to_complex()
    assert MAGIC_B602 == MAGIC_B60.to_complex()
    assert MAGIC_B662 == MAGIC_B66.to_complex()
    assert MAGIC_E62 == MAGIC_E6.to_complex()
    assert MAGIC_O62 == MAGIC_O6.to_complex()
    assert MAGIC_K62 == MAGIC_K6.to_complex()
    assert MAGIC_PHI2 == MAGIC_PHI.to_complex()

