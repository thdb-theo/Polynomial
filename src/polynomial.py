from numbers import Real, Number
from operator import itemgetter

from misc import reverseenum, iszero, round_floating_point_error, sqrt, cbrt, \
                 is_quite_close, sign
from math import cos, sin, acos, isinf, inf
from itertools import dropwhile, zip_longest
import re

e = 'abcdefghijklmnopqrstuvwxyz'
g = 'αβγδεζηθικλμνξοπρστυφχψω'
alphabet = e + g + e.upper() + g.upper()


class Polynomial:
    """TODO: Write documentations and doctests here
    TODO: Add Intergration"""
    presition = 5

    def __init__(self, *args, min_deg=None, **kwargs):
        coef_kwargs = {k: v for k, v in kwargs.items() if len(k) == 1}
        args = list(dropwhile(iszero, args))
        if not args:
            args = [0]  # if init '*args' only was 0s
        deg = len(args) - 1
        if deg > 100:
            raise OverflowError('The degree of the polynomial')
        if min_deg is not None and min_deg > deg:
            coef_kwargs[alphabet[min_deg]] = 0
        for k, v in coef_kwargs.items():
            deg = max(alphabet.find(k), deg)
        coeffs = coef_kwargs
        iter_args = iter(args)
        i = 0
        for s in alphabet:
            if s not in coeffs:
                try:
                    coeffs[s] = next(iter_args)
                except StopIteration:
                    coeffs[s] = 0
            if i == deg:
                break
            i += 1
        coeffs = dict(sorted(coeffs.items(), key=itemgetter(0)))
        coeffs = {k: round_floating_point_error(v) for k, v in coeffs.items()}
        for k, v in coeffs.items():
            setattr(self, k, v)
        assert self.a or len(coeffs) == 1
        self.coeffs = list(coeffs.values())
        if 'pres' in kwargs:
            self.presition = kwargs['pres']
        self.name = kwargs.get('name', 'f')

    @classmethod
    def from_roots(cls, *roots):
        r = 1
        for root in roots:
            r *= cls(1, -root)
        assert all([is_quite_close(abs(x), abs(y)) for x, y in zip(
            sorted(r.roots(), key=lambda x: x.real),
            sorted(roots, key=lambda x: x.real))]), (r, r.roots(), roots)
        return r

    def __setattr__(self, attr, value):
        self.__dict__[attr] = value
        if hasattr(self, 'coeffs') and len(attr) == 1:
            idx_alpha = alphabet.find(attr)
            if idx_alpha > self.degree:
                for a in alphabet[self.degree:idx_alpha]:
                    self.coeffs.append(0)
                    self.__dict__[a] = 0
                self.coeffs.append(value)
            else:
                self.coeffs[idx_alpha] = value

    def __getitem__(self, idx):
        return self.__dict__[alphabet[idx]]

    def __setitem__(self, idx, value):
        if idx < 0:
            idx += self.degree + 1
        setattr(self, alphabet[idx], value)

    def __call__(self, x):
        r = 0
        for i, c in reverseenum(self.coeffs, self.degree):
            r += c * x ** i
        return r

    def __add__(self, other):
        if isinstance(other, Number):
            other = Polynomial(other)
        if isinstance(other, (list, tuple)):
            other = Polynomial(*other)
        if isinstance(other, dict):
            other = Polynomial(**other)
        r = []
        dif_deg = self.degree - other.degree
        if dif_deg >= 0:
            coeffs1 = self.coeffs
            coeffs2 = [0] * dif_deg + list(other.coeffs)
        else:
            coeffs1 = [0] * -dif_deg + list(self.coeffs)
            coeffs2 = other.coeffs
        for c1, c2 in zip(coeffs1, coeffs2):
            r.append(c1 + c2)
        return Polynomial(*r)

    __radd__ = __add__

    def __sub__(self, other):
        return -other + self

    def __rsub__(self, other):
        return other + -self

    def __mul__(self, other):
        if isinstance(other, Real):
            other = Polynomial(other)
        if isinstance(other, (list, tuple)):
            other = Polynomial(*other)
        if isinstance(other, dict):
            other = Polynomial(**other)
        dif_deg = self.degree - other.degree
        if dif_deg >= 0:
            coeffs1 = self.coeffs
            coeffs2 = [0] * dif_deg + list(other.coeffs)
            highest_deg = self.degree

        else:
            coeffs1 = [0] * -dif_deg + list(self.coeffs)
            coeffs2 = other.coeffs
            highest_deg = other.degree
        q = []
        for deg1, c1 in reverseenum(coeffs1, highest_deg):
            for deg2, c2 in reverseenum(coeffs2, highest_deg):
                q.append([deg1+deg2, c1 * c2])

        e = dict(zip_longest((k for k, v in q), [0], fillvalue=0))
        for d, i in q:
            e[d] += i

        e = dropwhile(iszero, e.values())
        return Polynomial(*e)

    __rmul__ = __mul__

    def __pow__(self, p):
        """Interger powers of self
        Does not handle float or negative powers
        TODO: Implement nth roots"""
        assert isinstance(p, int) and p >= 0
        r = Polynomial(1)
        for _ in range(p):
            r *= self
        return r

    def __truediv__(self, other):
        """Fast Polynomial division by using Extended Synthetic Division.
        Also works with non-monic polynomials.
        Source: https://en.wikipedia.org/wiki/Synthetic_division#Python_implementation"""
        if isinstance(other, Real):
            return self._truediv_by_number(other)
        if isinstance(other, (list, tuple)):
            other = Polynomial(*other)
        if isinstance(other, dict):
            other = Polynomial(**other)
        if other.degree == 0:
            return self._truediv_by_number(other.a)
        dividend = self.coeffs
        divisor = other.coeffs
        out = list(dividend)
        normalizer = divisor[0]
        for i in range(len(dividend)-(len(divisor)-1)):
            out[i] /= normalizer
            coef = out[i]
            if coef != 0:
                for j in range(1, len(divisor)):
                    out[i + j] += -divisor[j] * coef
        separator = -len(divisor) + 1
        return Polynomial(*out[:separator]), out[separator:]

    def _truediv_by_number(self, other):
        return Polynomial(*(i / other for i in self.coeffs))

    def __repr__(self):
        r = ['%s(x) = ' % self.name]
        for i, c in reverseenum(self.coeffs, self.degree):
            if isinstance(c, Real):
                r.append('%sx^%s + ' % (round(c, self.presition), i))
            else:
                r.append('%sx^%s + ' % (c, i))

        s = ''.join(r)
        s = re.sub(r' \+ (0|0\.0|-0\.0)x\^\d+', '', s)
        s = re.sub(r'x\^0', '', s)
        s = re.sub(r'x\^1 ', 'x ', s)
        s = re.sub(r'-(1|1\.0)x', '-x', s)
        s = re.sub(r' (1|1\.0)x', ' x', s)
        s = re.sub(r'\+ -', '- ', s)
        s = re.sub(r'\.0 ', ' ', s)
        s = re.sub(r' \+ $', '', s)
        return s

    __str__ = __repr__

    def short_str(self):
        return re.sub(r'%s\(x\) = ' % self.name, '', str(self))

    def __neg__(self):
        """Polymails with all coeffs *-1"""
        return self * -1

    def __pos__(self):
        return self + 0.

    __copy__ = __pos__

    def __bool__(self):
        return any(lambda x: not iszero(x), self.coeffs)

    def __eq__(self, other):
        return not self - other

    def tangent(self, x):
        """Equation of tangent line at f(x)"""
        a = self.derivative()(x)
        b = a * -x + self(x)
        return Polynomial(a, b)

    def inflection_points(self, imag=True):
        """Inflection points
        Where f(x) goes from concave to convex and vice versa
        f''(x) = 0"""
        second_deriv = self.derivative().derivative()
        return [(x, self(x)) for x in second_deriv.roots(imag=imag)]

    def inflection_tangents(self, imag=True):
        """Equations of tangent at inflection points"""
        return [self.tangent(x) for x, y in self.inflection_points(imag=imag)]

    @property
    def degree(self):
        """Degree of Polynomial; The power of the coeff with the highest power"""
        return len(self.coeffs) - 1

    def factor(self):
        """Factorise equation
        Example:
        ---------
        Polynomial(1, 1, -2).factor()
        '(x+2)(x-1)'
        Polynomial(0.5, -2, 3, -2).factor()
        '(x-2.0)(0.5x^2 + -1.0x + 1.0)'"""
        roots = self.roots(imag=True)
        factors = []
        if any(i for i in roots if isinstance(i, complex)):
            roots_iter = iter(roots)
            root = next(roots_iter)
            quotient = self
            prev_quo = quotient
            while True:
                quotient, remainder = (quotient / Polynomial(1, -root))
                if not iszero(remainder[0]):
                    factors.append('(%s)' % prev_quo.short_str())
                    break
                factors.append('(x-%s)' % root)
                if quotient.degree == 0:
                    break
                prev_quo = quotient
                try:
                    root = next(roots_iter)
                except StopIteration:
                    factors.append('(%s)' % prev_quo.short_str())
                    break
        elif roots:
            factors.append('%s' % self.a if self.a != 0 else '')
            for root in roots:
                factors.append('(x-%s)' % root)
        else:
            factors.append(self.short_str())
        asstr = ''.join(factors)
        asstr = re.sub(r'(\+|-)+(0|0\.0)\)', ')', asstr)
        asstr = re.sub(r'^(1|1\.0)\(', '(', asstr)
        asstr = re.sub(r'--', '+', asstr)
        asstr = re.sub(r'\((1|1.0)x', '(x', asstr)
        for n in re.findall(r'(?!\^)\d+\.\d+', asstr):
            asstr = asstr.replace(n, str(round(float(n), self.presition)))
        split = re.sub(r'\)\(', ')SPLIT(', asstr)
        factor_list = split.split('SPLIT')
        for i, factor in enumerate(factor_list):
            eq_terms = factor_list.count(factor)
            if eq_terms > 1:
                factor_list[i] = factor + '^%s' % eq_terms
                factor_list = [j for j in factor_list if j != factor]
        return ''.join(factor_list)

    def extremes(self, imag=True):
        deriv = self.derivative()
        return deriv.roots(imag=imag)

    def derivative(self):
        r = []
        for i, c in reverseenum(self.coeffs[:-1], self.degree):
            r.append(i * c)
        if not r:
            r = [0]
        return Polynomial(*r)

    def nth_derivative(self, n):
        r = self
        for _ in range(n):
            r = r.derivative()
        return r

    def plot(self, start=None, end=None):
        from numpy import linspace, array
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        roots = self.roots(imag=False)
        if not roots:
            start, end = -5, 5
        else:
            if start is None:
                start = int(min(roots)) - 1
            if end is None:
                end = int(max(roots)) + 1
        x = linspace(start, end, 1000)
        curve = sum(a*x**i for i, a in reverseenum(self.coeffs, self.degree))
        deriv = self.derivative()
        deriv_curve = deriv(x)
        deriv_x = ax.plot(x, deriv_curve, label='%s\'(x) = %s' % (self.name, deriv.short_str()))
        tangents = self.inflection_tangents(imag=False)
        ts = []
        for t in tangents:
            line = t(x)
            ts.append(ax.plot(x, line, color=[.2, .3, .7], label='tangent at f\'\'(x)=0'))
        points = array([(i, self(i)) for i in self.extremes(imag=False)] +
                       self.inflection_points(imag=False) + [(i, 0.) for i in roots])
        f_x = ax.plot(x, curve, label=str(self))
        p = ax.scatter(*points.T)
        for x, y in points:
            ax.text(x, y, '{0:.2}, {1:.2}'.format(float(x), float(y)))
        ax.set_title('f(x)')
        if ts:
            plt.legend(handles=[p, f_x[0], deriv_x[0], *ts[0]])
        else:
            plt.legend(handles=[f_x[0], deriv_x[0]])
        plt.grid(b=True)
        plt.show()

    def roots(self, imag=True, **kwargs):
        methods = [self._constant_root, self._linear_root,
                   self._quad_roots, self._cubic_roots,
                   self._quartic_roots]
        if self.degree < 5:
            roots = methods[self.degree](**kwargs)
        else:
            roots = self._newton_roots(**kwargs)
        mapped_roots = list(map(round_floating_point_error, roots))
        if imag:
            return mapped_roots
        else:
            return [x.real for x in mapped_roots if iszero(x.imag)]

    def _constant_root(self, **kwargs):
        return []

    def _linear_root(self, **kwargs):
        return [-self.b / self.a]

    def _quad_roots(self, **kwargs):
        a, b, c = self.a, self.b, self.c
        x1 = (-b - sqrt(b**2 - 4 * a * c)) / (2 * a)
        x2 = (-b + sqrt(b**2 - 4 * a * c)) / (2 * a)
        return [x1, x2]

    def _cubic_roots(self, **kwargs):
        a, b, c, d = self.coeffs
        if any(isinstance(i, complex) for i in self.coeffs):
            return self._cubic_roots_complex_coef(**kwargs)
        f = (3 * c / a - b**2 / a**2) / 3
        g = (2 * b**3 / a**3 - 9 * b * c / a**2 + 27 * d / a) / 27
        h = g**2 / 4 + f**3 / 27
        if f == g == h == 0.0:
            return [cbrt(d / a) * -1] * 3
        elif h.real <= 0:
            i = sqrt(g**2 / 4 - h)
            j = cbrt(i)
            K = acos(-g / (2 * i))
            L = j * -1
            M = cos(K / 3)
            N = sqrt(3) * sin(K / 3)
            P = (b / (3 * a)) * -1
            x1 = 2 * j * cos(K / 3) - (b / (3 * a))
            x2 = L * (M + N) + P
            x3 = L * (M - N) + P
            return [x1, x2, x3]
        elif h.real > 0:
            R = -g / 2 + sqrt(h)
            S = cbrt(R)
            T = -g / 2 - sqrt(h)
            U = cbrt(T)
            i = 1j
            realx1 = S + U - b / (3*a)
            imagx2 = -(S + U) / 2 - b / (3 * a) + i * (S - U) * sqrt(3) / 2
            imagx3 = -(S + U) / 2 - b / (3 * a) - i * (S - U) * sqrt(3) / 2
            return [realx1, imagx2, imagx3]

    def _cubic_roots_complex_coef(self, **kwargs):
        f = self / self.a
        p, q, r = f.b, f.c, f.d
        A = ((cbrt(-2*p**3 + 9*p*q - 27*r + 3*sqrt(3) *
                   sqrt(-p**2*q**2 + 4*q**3 + 4*p**3*r - 18*p*q*r + 27*r**2))) /
             (3*cbrt(2)))
        if iszero(A, e=1e-5):
            if all(isinstance(i, Real) for i in f.coeffs):
                x1 = f._cubic_roots(**kwargs)
            else:
                x1 = f.newtons_method(**kwargs)
            g, remainder = f / Polynomial(1, -x1)
            assert iszero(remainder[0])
            return [x1] + g.roots()

        B = (-p**2 + 3*q)/(9*A)
        x1 = -p/3 + A - B
        x2 = -p/3 + ((-1 - 1j*sqrt(3))/2) * A - ((-1 + 1j*sqrt(3))/2) * B
        x3 = -p/3 + ((-1 + 1j*sqrt(3))/2) * A - ((-1 - 1j*sqrt(3))/2) * B
        return [x1, x2, x3]

    def _quartic_roots(self, **kwargs):
        a, b, c, d, e = [u / self.a for u in self.coeffs]
        f = c - 3*b**2/8
        g = d + b**3 / 8 - b * c / 2
        h = e - 3*b**4 / 256 + b ** 2 * c / 16 - b*d/4
        eq = Polynomial(1, f/2, (f**2 - 4 * h)/16, -g**2/64)
        roots = eq.roots(**kwargs)
        try:
            Y1, Y2 = [i for i in roots if not iszero(i)][:2]
        except ValueError:
            x1 = self.newtons_method(**kwargs)
            quo, remainder = self / Polynomial(1, -x1)
            assert iszero(remainder[0], e=kwargs.get('e', 1e-6)), remainder[0]
            return [x1] + quo.roots(**kwargs)
        p = sqrt(Y1)
        q = sqrt(Y2)
        r = -g/(8*p*q)
        s = b/(4*a)
        x1 = p + q + r - s
        x2 = p - q - r - s
        x3 = -p + q - r - s
        x4 = -p - q + r - s
        return [x1, x2, x3, x4]

    def _newton_roots(self, **kwargs):
        e = kwargs.get('e', 1e-6)
        r = []
        poly = self
        while True:
            if poly.degree <= 4:
                if all(isinstance(i, Real) for i in poly.coeffs) or poly.degree == 0:
                    return r + poly.roots(**kwargs)
            root = poly.newtons_method(**kwargs)
            r.append(root)
            poly, remainder = poly / Polynomial(1, -root)
            assert iszero(remainder[0], e=e), (remainder[0], root)

    def newtons_method(self, **kwargs):
        x0, e = kwargs.get('x0', 0), kwargs.get('e', 1e-6)
        max_iter = kwargs.get('max_iter', 1e6)

        def dx(x):
            return abs(0-self(x))

        df = self.derivative()
        while df(x0) == 0:
            x0 += 1
        delta = dx(x0)
        iterations = 0
        switch_to_complex = int(max_iter * 0.5)
        while delta > e and iterations != max_iter:
            x0 -= self(x0)/df(x0)
            delta = dx(x0)
            if iterations == switch_to_complex:
                x0 = 1+1j
                while df(x0) == 0:
                    x0 += 1+1j
            iterations += 1
        if iterations == max_iter:
            return None
        return x0

    def limit(self, x):
        if not isinf(x):
            return self(x)
        if self.degree == 0:
            return self.a
        if x > 0:
            return inf * sign(self.a)
        return inf * -(self.degree % 2 - 0.5)


def plot(p):
    p.plot()


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    P = Polynomial
    for i in range(-2, 2):
        for j in range(-2, 2):
            for k in range(-2, 2):
                for l in range(-2, 2):
                    for m in range(-2, 2):
                        try:
                            P(i, j, k, l, m).roots()
                        except Exception as e:
                            print(i, j, k, l)
                            raise e

