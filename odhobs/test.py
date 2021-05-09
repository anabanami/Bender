import numpy as np
from scipy.special import eval_hermite, factorial
from ho import psi as cpsi

# Python implementation. Adapted from
# https://www.numbercrunch.de/blog/2014/08/calculating-the-hermite-functions/ with
# wisdom from https://scicomp.stackexchange.com/questions/30896/
#
# I managed to get a ~50% speedup by using frexp to do the rescaling using the base-2
# exponent the floats already have, without actually calculating a log. This also avoids
# taking a log of zero (discussed in the stackexchange post), which is otherwise awkward
# to do.

def psi(n, x):
    h_prev = np.ones_like(x) * np.pi ** -0.25
    h = np.sqrt(2.0) * x * np.pi ** -0.25
    sum_log_scale = np.zeros_like(x)
    if n == 0:
        h = h_prev
    for i in range(2, n + 1):
        h, h_prev = np.sqrt(2 / i) * x * h - np.sqrt((i - 1) / i) * h_prev, h
        _, log_scale = np.frexp(h)
        scale = np.exp2(-log_scale)
        h *= scale
        h_prev *= scale
        sum_log_scale += log_scale
    return h * np.exp(-(x ** 2) / 2 + np.log(2) * sum_log_scale)


def psi_explicit(n, x):
    c = 1 / (np.pi ** 0.25 * np.sqrt(2 ** n * factorial(n)))
    return c * eval_hermite(n, x) * np.exp(-(x ** 2) / 2)

import time

n_small = 9
n_big = 1000

x = np.linspace(-10, 10, 1024, endpoint=False)

# Quick test for correctness
print("testing correctness of first ten states")
for n in range(10):
    a = psi(n, x)
    b = psi_explicit(n, x)
    c = cpsi(n, x)
    assert np.allclose(a, c)
    assert np.allclose(b, c)

print('testing handling of dtypes')
a = psi(n_small, 10)
print("  np float")
c = cpsi(n_small, np.array([10.0])[0])
assert a == c

print("  np 0d float arr")
c = cpsi(n_small, np.array(10.0))
assert a == c

print("  np int")
c = cpsi(n_small, np.array([10])[0])
assert a == c

print("  np 0d int arr")
c = cpsi(n_small, np.array(10))
assert a == c

print("  pyint")
c = cpsi(n_small, 10)
assert a == c

print("  pyfloat")
c = cpsi(n_small, 10.0)
assert a == c

print(f"testing array speed with {x.shape} array")

start_time = time.time()
a = psi(n_big, x)
pytime = time.time() - start_time
print(f'  pytime: {pytime}')

start_time = time.time()
c = cpsi(n_big, x)
cytime = time.time() - start_time
print(f'  cytime: {cytime}')

assert np.allclose(a, c)
print(f'  {(pytime / cytime):.1f}× speedup')


print(f"testing python loop speed with {x.size} points")

a = np.empty_like(x)
start_time = time.time()
for i, x_i in enumerate(x.flat):
    a.flat[i] = psi(n_big, x_i)
pytime = time.time() - start_time
print(f'  pytime: {pytime}')

c = np.empty_like(x)
start_time = time.time()
for i, x_i in enumerate(x.flat):
    c.flat[i] = cpsi(n_big, x_i)
cytime = time.time() - start_time
print(f'  cytime: {cytime}')

assert np.allclose(a, c)
print(f'  {(pytime / cytime):.1f}× speedup')
