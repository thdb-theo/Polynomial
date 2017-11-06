from numbers import Real
from math import copysign, isclose
from cmath import exp, log
from types import GeneratorType


def cbrt(x):
    """cube root of x; does not return negative numbers
    Examples:
    ------------
    >>> cbrt(-8)
    -2.0
    >>> cbrt(8)
    2.0
    """

    if isinstance(x, Real):
        return copysign(pow(abs(x), 1/3), x.real)
    elif isinstance(x, complex):
        return exp((1/3)*log(x))
    else:
        raise TypeError('x must be a number')


def sqrt(x):
    """Square root of x; support complex numbers
    Examples:
    -------------
    >>> sqrt(4)
    2.0
    >>> sqrt(-1)
    1j
    >>> sqrt(0)
    0.0"""
    a = pow(x, 0.5)
    if isinstance(x, complex):
        return a
    elif x < 0:
        return complex(0, a.imag)
    else:
        return a.real


def iszero(x, e=1e-7):
    """Return -1e-7 < x < 1e-7
    in other words if x is zero or very close to zero
    Also works with complex numbers
    Examples:
    ----------
    >>> assert iszero(0)
    >>> assert iszero(1e-7) and iszero(-1e-7) and iszero(1e-10)
    >>> assert iszero(0.0000001)
    >>> assert not iszero(0.000001)
    >>> assert not iszero(complex(0, 1)) and iszero(complex(0, 1e-10))
    """
    return isclose(abs(x), 0.0, abs_tol=e)


def is_quite_close(x, y, e=0.1):
    return isclose(x, y, abs_tol=e)


def filter_near0imag(x):
    """return only the real part of a complex number whos imaginary part
    is zero or close to zero
    Example:
    ---------
    >>> filter_near0imag(complex(1, 1e-10))
    1.0
    >>> filter_near0imag(complex(1, 1))
    (1+1j)
    """
    if not isinstance(x, complex):
        return x
    if isclose(x.imag, 0, abs_tol=1e-8):
        return x.real
    return x


def round_floating_point_error(x):
    """return x rounded to 5 digits if round(x,5) is a multiple of 0.5
    else return x
    Example:
    ----------
    >>> round_floating_point_error(0.999999)
    1.0
    >>> round_floating_point_error(0.2)
    0.2"""
    x = filter_near0imag(x)
    if isinstance(x, complex):
        return complex(round_floating_point_error(x.real),
                       round_floating_point_error(x.imag))
    if isinstance(x, str):
        print(x)
        raise ValueError
    a = round(x, 5)
    if a % 0.5 == 0:
        return a
    else:
        return x


def reverseenum(seq, start=None):
    """Reversed enumerate
    if i is None and x is a generator the generator is made into a
    list to find len and then looped through again
    Params:
    ---------
    seq: An iterable
    start: the start of the numeration
    """
    if start is None:
        if isinstance(seq, GeneratorType):
            seq = list(seq)
        start = len(seq)
    for n in seq:
        yield start, n
        start -= 1


def sign(x):
    return (x > 0) - (x < 0)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    print(sign(0))