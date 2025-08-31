import numpy as np
from .core import Component


class Source(Component):
    def __init__(self, mode:int=0):
        super().__init__(name="Source", params={"mode": mode})

    def prepare(self):
        mode = int(self.params.get("mode", 0))
        if mode == 0:
            return np.array([1.0+0j, 0.0+0j])
        else:
            return np.array([0.0+0j, 1.0+0j])


class PhaseShifter(Component):
    def __init__(self, phi: float, mode: int):
        super().__init__(name="PhaseShifter", params={"phi": phi, "mode": mode})

    def unitary(self):
        phi = float(self.params["phi"])
        mode = int(self.params["mode"])
        U = np.eye(2, dtype=complex)
        if mode == 0:
            U[0,0] = np.exp(1j*phi)
        else:
            U[1,1] = np.exp(1j*phi)
        return U


class BeamSplitter(Component):
    def __init__(self, theta: float, phi: float=0.0):
        super().__init__(name="BeamSplitter", params={"theta": theta, "phi": phi})

    def unitary(self):
        theta = float(self.params["theta"]) # mixing angle
        phi = float(self.params.get("phi", 0.0))
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([
            [c, np.exp(1j*phi)*s],
            [-np.exp(-1j*phi)*s, c]
        ], dtype=complex)


class Detector(Component):
    def __init__(self):
        super().__init__(name="Detector")

    def measure(self, state):
        probs = np.abs(state)**2
        return {
            "p(mode0)": float(probs[0].real),
            "p(mode1)": float(probs[1].real),
        }