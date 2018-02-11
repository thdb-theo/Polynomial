from numbers import Number
import math
from functools import reduce
import random
import re

from polynomial import Polynomial
import misc


class Rational:
    __slots__ = 'numerator', 'denominator', 'name'

    def __init__(self, numerator, denominator, name='f'):
        if isinstance(numerator, (tuple, list)):
            numerator = Polynomial(*numerator)
        if isinstance(denominator, (tuple, list)):
            denominator = Polynomial(*denominator)
        if isinstance(denominator, Number):
            numerator = Polynomial(numerator)
        if isinstance(denominator, Number):
            denominator = Polynomial(denominator)
        try:
            n2int = list(map(misc.int_if_mod1, numerator.coeffs))
            d2int = list(map(misc.int_if_mod1, denominator.coeffs))
            gcd_coefs = reduce(math.gcd, n2int + d2int)
            numerator, remainder = numerator / gcd_coefs
            assert not remainder
            denominator, remainder = denominator / gcd_coefs
            assert not remainder
        except TypeError:  # float in coeffs
            pass
        if not any((numerator / denominator)[1]):  # No remainder
            numerator /= denominator
            denominator = Polynomial(1)
        n_factors = numerator.factors()
        d_factors = denominator.factors()
        for factor in set(n_factors) & set(d_factors):
            numerator /= factor
            denominator /= factor
        gcd_x = min(numerator.lowest_nonzero_coef(), denominator.lowest_nonzero_coef())
        numerator /= Polynomial(1, min_deg=gcd_x)
        denominator /= Polynomial(1, min_deg=gcd_x)
        self.numerator = numerator
        self.denominator = denominator
        self.name = name

    def __str__(self):
        num_str = self.numerator.short_str()
        den_str = self.denominator.short_str()
        longest = max(num_str, den_str, key=len)
        top = ' ' * 7 + num_str.center(len(longest))
        bar = 'f(x) = ' + '-' * (len(longest))
        bottom = ' ' * 7 + den_str.center(len(longest))
        return '\n'.join([top, bar, bottom])

    def short_str(self):
        return str(self)

    @classmethod
    def from_string(cls, string):
        formated_string = re.sub('[()]', '', string)
        numerator, denominator = formated_string.split('/')
        return cls(Polynomial.from_string(numerator), Polynomial.from_string(denominator))

    __repr__ = __str__

    def copyable(self):
        return '(%s)/(%s)' % (self.numerator.short_str(), self.denominator.short_str())

    def __call__(self, x):
        return self.numerator(x) / self.denominator(x)

    def __neg__(self):
        return Rational(-self.numerator, self.denominator)

    def __mul__(self, other):
        if isinstance(other, (Polynomial, Number)):
            return Rational(self.numerator * other, self.denominator)
        elif isinstance(other, Rational):
            return Rational(self.numerator * other.numerator,
                            self.denominator * other.denominator)
        else:
            return ValueError

    def __add__(self, other):
        if isinstance(other, Rational):
            return Rational(self.numerator * other.denominator + other.numerator * self.denominator,
                            self.denominator * other.denominator)
        else:
            return Rational(self.numerator + other * self.denominator, self.denominator)

    def __sub__(self, other):
        if isinstance(other, Rational):
            return Rational(self.numerator * other.denominator - other.numerator * self.denominator,
                            self.denominator * other.denominator)
        else:
            return Rational(self.numerator - other * self.denominator, self.denominator)

    def __truediv__(self, other):
        if isinstance(other, Rational):
            return self * Rational(other.denominator, other.numerator)
        else:
            return Rational(self.numerator, other * self.denominator)

    def __pow__(self, p):
        """Interger powers of self
        Does not handle float or negative powers
        TODO: Implement nth roots"""
        assert isinstance(p, int) and p >= 0
        r = Rational(Polynomial(1), Polynomial(1))
        for _ in range(p):
            r *= self
        return r

    def roots(self, **kwargs):
        numerator_roots = self.numerator.roots(**kwargs)

        def filter_0_in_den(x):
            return not misc.iszero(self.denominator(x), kwargs.get('e', 1e-10))
        return list(filter(filter_0_in_den, numerator_roots))

    def derivate(self):
        return Rational(self.numerator.derivate() * self.denominator -
                        self.numerator * self.denominator.derivate(),
                        self.denominator ** 2)

    nth_derivate = Polynomial.nth_derivate  # This is a really bad idea

    def horizontal_asymptote(self):
        if self.numerator.degree == self.denominator.degree:
            return Polynomial(self.numerator[0] / self.denominator[0])
        if self.numerator.degree > self.denominator.degree:
            return (self.numerator / self.denominator)[0]
        if self.numerator.degree < self.denominator.degree:
            return Polynomial(0)
        else:
            return None

    def vertical_asymptotes(self):
        return {'x=%s' % r for r in self.denominator.roots(imag=False)}

    def asymptotes(self):
        asymps = self.vertical_asymptotes()
        h_as = self.horizontal_asymptote()
        if h_as is not None:
            asymps.add(h_as)
        return asymps

    @classmethod
    def random(cls, max_deg=4, range_=(-5, 5)):
        num_deg = random.randint(0, max_deg)
        den_deg = random.randint(1, max_deg)
        num_coefs = [random.randint(*range_) for _ in range(num_deg + 1)]
        den_coefs = [random.randint(*range_) for _ in range(den_deg + 1)]
        return cls(num_coefs, den_coefs)

    def plot(self, start=-15, end=15):
        from numpy import linspace
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        roots = self.roots()
        if roots:
            if start is None:
                start = int(min(roots)) - 1
            if end is None:
                end = int(max(roots)) + 1
        ax.set_ylim([start, end])
        ax.set_xlim([start, end])
        x = linspace(start, end, 1000)
        curve = self(x)
        f_x = ax.plot(x, curve, color=[0., 0., 1.], label=str(self))
        dx = self.derivate()
        dxcurve = dx(x)
        df_x = ax.plot(x, dxcurve, color=[1., 0., 0.], label=str(dx).replace('f(x)', 'f\'(x)'))
        for asym in self.asymptotes():
            if isinstance(asym, str):
                line = plt.axvline(x=float(asym.lstrip('x=')), color=[0., 1., 0.])
                line.set_dashes([2, 4])
            else:
                line = asym(x)
                h = ax.plot(x, line, color=[0., 1., 0.])
                h[0].set_dashes([3, 4])
        L = plt.legend(handles=[f_x[0], df_x[0]])
        plt.setp(L.texts, family='Consolas')
        ax.set_title(str(self), fontname='Consolas')
        plt.grid(b=True)
        plt.show()


if __name__ == '__main__':
    P = Polynomial
    R = Rational
    f = R(P(1, 0, -2, 4), P(9, 0, -2))
