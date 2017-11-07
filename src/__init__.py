"""Polynomials
Author: Theodor Tollersrud
Thanks to:
www.wikipedia.org for polynomial division
https://en.wikipedia.org/wiki/Synthetic_division#Python_implementation
www.1728.org for root of cubic and quartic functions:
http://www.1728.org/cubic2.htm, http://www.1728.org/quartic2.htm
www.danielhomola.com for newton's method
http://danielhomola.com/2016/02/09/newtons-method-with-10-lines-of-python/
"""

__version__ = 0.1

import sys

from polynomial import Polynomial, plot, alphabet

cmdargs = sys.argv[1:]

P = Polynomial
poly = Polynomial

if __name__ == '__main__':
    f = P(*map(float, cmdargs))