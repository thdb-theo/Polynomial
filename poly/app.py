import sys
import re

from PyQt4 import QtGui, QtCore

from polynomial import Polynomial
from rational import Rational


class Window(QtGui.QMainWindow):
    width, height = 420, 130

    def __init__(self):
        super().__init__()
        self.setFixedSize(Window.width, Window.height) 
        self.setWindowTitle('Find Roots')
        self.setWindowIcon(QtGui.QIcon('Images/roots.png'))
        self.poly = None
        self.setFont(QtGui.QFont('Times New Roman'))
        self.home()

    def home(self):
        self.is_imag = True
        self.imag_b = QtGui.QCheckBox('Return imaginary numbers?')
        self.imag_b.adjustSize()
        self.imag_b.setParent(self)
        self.imag_b.toggle()
        self.imag_b.move(10, 5)
        self.imag_b.stateChanged.connect(self.toggle_imag)

        self.instruction = QtGui.QLabel(self)
        self.instruction.setText('Enter coefficients of a polynomial seperated by commas.')
        self.instruction.move(10, 35)
        self.instruction.adjustSize()

        self.text = QtGui.QLabel(self)

        self.entry = QtGui.QLineEdit(self)
        self.entry.returnPressed.connect(self.find_roots)
        self.entry.move(10, 60)
        self.entry.resize(400, 30)

        self.confirm = QtGui.QPushButton('Find Roots!', self)
        self.confirm.move(10, 100)
        self.confirm.clicked.connect(self.find_roots)

        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Return), self, self.find_roots)

        self.plot_b = QtGui.QPushButton('Plot', self)
        self.plot_b.clicked.connect(self.plot)
        self.plot_b.move(120, 100)

        self.factor_b = QtGui.QPushButton('Factorise', self)
        self.factor_b.clicked.connect(self.factor)
        self.factor_b.move(230, 100)

        self.derivate_b = QtGui.QPushButton('Derivate', self)
        self.derivate_b.clicked.connect(self.derivate)
        self.derivate_b.move(340, 100)

        self.eq = QtGui.QLabel(self)
        self.eq.move(10, Window.height)

        self.show()

    def toggle_imag(self):
        self.is_imag = not self.is_imag

    def find_roots(self):
        self.entry_text = self.entry.text()
        try:
            self.poly = self.get_poly(self.entry_text)
        except ValueError:
            QtGui.QMessageBox.warning(self, 'warning', 'Invalid arguments')
            return

        roots = self.poly.roots(imag=self.is_imag)
        self.eq.setFont(QtGui.QFont('Consolas', 8))
        s = '%s = 0' % self.poly.short_str()
        self.eq.setText(re.sub("(.{44})", "\\1\n",
                               s, 0, re.DOTALL))
        self.eq.adjustSize()
        t = []
        for i, r in enumerate(roots):
            t.append('x<sub>%s</sub> = %s' % (i, r))
        s = '<br>'.join(t)
        self.text.setText(s)
        self.text.adjustSize()

        self.text.move(10, Window.height + self.eq.height())
        new_height = Window.height + self.eq.height() + self.text.height() + 10
        self.setFixedSize(Window.width, new_height)

    def plot(self) -> None:
        self.entry_text = self.entry.text()
        try:
            self.poly = self.get_poly(self.entry_text)
        except ValueError:
            QtGui.QMessageBox.warning(self, 'warning', 'Invalid arguments')
            return
        self.poly.plot()

    def factor(self):
        self.entry_text = self.entry.text()
        try:
            self.poly = self.get_poly(self.entry_text)
        except ValueError:
            QtGui.QMessageBox.warning(self, 'warning', 'Invalid arguments')
            return
        self.eq.setText('')
        self.text.setText(self.poly.factor())
        self.text.move(10, Window.height)
        self.text.adjustSize()
        self.text.setWordWrap(True)
        self.setFixedSize(Window.width, Window.height + self.text.height())

    def derivate(self):
        self.entry_text = self.entry.text()
        try:
            self.poly = self.get_poly(self.entry_text)
        except ValueError:
            QtGui.QMessageBox.warning(self, 'warning', 'Invalid arguments')
            return
        self.eq.setText('')
        self.text.setText(str(self.poly.derivate()))
        self.text.setFont(QtGui.QFont('Courier'))
        self.text.move(10, Window.height)
        self.text.adjustSize()
        self.text.setWordWrap(True)
        self.setFixedSize(Window.width, Window.height + self.text.height())

    @staticmethod
    def get_poly(text):
        if 'x' in text:
            return Polynomial.from_string(text)
        terms = re.findall(r'-?\d+\.?\d*|/', text)
        if '/' in terms:
            numerator, denominator = terms[:terms.index('/')], terms[terms.index('/') + 1:]
            num_coefs, den_coefs = list(map(float, numerator)), list(map(float, denominator))
            return Rational(num_coefs, den_coefs)
        else:
            coefs = map(float, terms)
            return Polynomial(*coefs)
def main():
    app = QtGui.QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
