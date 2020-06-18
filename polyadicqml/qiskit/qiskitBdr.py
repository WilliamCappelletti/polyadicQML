"""Circuit builders for the qiskit/IBMQ interface"""
from qiskit import QuantumRegister, QuantumCircuit
from numpy import pi

from ..circuitBuilder import circuitBuilder

class __qiskitGeneralBuilder__(circuitBuilder):
    """Abstract class for Qiskit-circuits builders. 
    """
    def __init__(self, nbqbits, *args, **kwargs):
        super().__init__(nbqbits)
        self.qr = QuantumRegister(self.nbqbits, 'qr')
        self.qc = QuantumCircuit(self.qr)

    def circuit(self):
        """Return the built circuit.

        Returns
        -------
        qiskit.QuantumCircuit
        """
        return self.qc

    def measure_all(self):
        self.qc.measure_all()


class qiskitBuilder(__qiskitGeneralBuilder__):
    """Qiskit-circuits builder using rx, rz and cz gates.
    """
    def __init__(self, nbqbits, *args, **kwargs):
        super().__init__(nbqbits)
    
    def alldiam(self, idx=None):
        if idx is None:
            self.qc.rx(pi/2, self.qr)
        else:
            for i in idx:
                self.qc.rx(pi/2, self.qr[i])

    def input(self, idx, theta):
        if isinstance(idx, list):
            for p, i in enumerate(idx):
                self.qc.rz(theta[p], self.qr[i])
                self.qc.rx(pi/2, self.qr[i])
        else:
            self.qc.rz(theta, self.qr[idx])
            self.qc.rx(pi/2, self.qr[idx])

    def allin(self, x):
        for i, qb in enumerate(self.qr):
            self.qc.rz(x[i], qb)

        self.qc.rx(pi/2, self.qr)

    def cc(self, a, b):
        self.qc.cz(self.qr[a], self.qr[b])
        self.qc.rx(pi/2, [self.qr[a], self.qr[b]])

class ibmqNativeBuilder(__qiskitGeneralBuilder__):
    """Qiskit-circuits builder using IBMQ native gates (u1, u2, and cx).
    """
    def __init__(self, nbqbits, *args, **kwargs):
        super().__init__(nbqbits)

    def alldiam(self, idx=None):
        if idx is None:
            self.qc.u2(-pi/2, pi/2, self.qr)
        else:
            for i in idx:
                self.qc.u2(-pi/2, pi/2, self.qr[i])

    def input(self, idx, theta):
        if isinstance(idx, list):
            for p, i in enumerate(idx):
                self.qc.u1(theta[p], self.qr[i])
                self.qc.u2(-pi/2, pi/2, self.qr[i])
        else:
            self.qc.u1(theta, self.qr[idx])
            self.qc.u2(-pi/2, pi/2, self.qr[idx])

    def allin(self, x):
        for i, qb in enumerate(self.qr):
            self.qc.u1(x[i], qb)

        self.qc.u2(-pi/2, pi/2, self.qr)

    def cc(self, a, b):
        self.qc.u2(-pi/2, pi/2, self.qr[b])
        self.qc.u1(pi/2, self.qr[b])
        self.qc.cx(self.qr[a], self.qr[b])
        self.qc.u1(-pi/2, self.qr[b])
        self.qc.u2(-pi/2, pi/2, self.qr[a])