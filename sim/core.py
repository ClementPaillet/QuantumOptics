from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any


Array = np.ndarray


@dataclass
class Component:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    def unitary(self) -> Array:
        """Return 2x2 unitary for single-photon dual-rail encoding.
        Sources/detectors override behavior as needed.
        """
        return np.eye(2, dtype=complex)

    def prepare(self) -> Array | None:
        """For sources: return a normalized 2-vector state, else None."""
        return None

    def measure(self, state: Array) -> Dict[str, float] | None:
        """For detectors: return measurement results, else None."""
        return None


@dataclass
class Circuit:
    components: List[Component] = field(default_factory=list)

    def add(self, comp: Component) -> None:
        self.components.append(comp)


    def simulate(self) -> Dict[str, Any]:
        # 1) Find first source
        state = None
        for c in self.components:
            s = c.prepare()
            if s is not None:
                state = s
                break
            if state is None:
                # default source: photon in mode 0
                state = np.array([1.0 + 0j, 0.0 + 0j])

        # 2) Apply unitaries in order
        U = np.eye(2, dtype=complex)
        for c in self.components:
            Uc = c.unitary()
            if Uc is not None:
                U = Uc @ U

        # 3) Final state
        psi_out = U @ state

        # 4) If any detector present, use last one to produce results
        results = None
        for c in self.components[::-1]:
            m = c.measure(psi_out)
            if m is not None:
                results = m
                break

        if results is None:
            # Default: probabilities per output mode
            probs = np.abs(psi_out) ** 2
            results = {
                "p(mode0)": float(probs[0].real),
                "p(mode1)": float(probs[1].real),
            }


        return {
            "state_out": psi_out,
            "unitary": U,
            "results": results,
        }