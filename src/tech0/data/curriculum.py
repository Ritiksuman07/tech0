"""Phased data curriculum from the design doc.

Phase 1 foundation : 25% target / 75% general
Phase 2 annealing   : 70% target / 30% general
Phase 3 SFT         : 65% target / 35% general
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PhasePlan:
    name: str
    target_ratio: float
    general_ratio: float
    tokens: int

    @property
    def ratio_sum(self) -> float:
        return self.target_ratio + self.general_ratio


class DataMixer:
    def __init__(self, curriculum: dict):
        self.phases: dict[str, PhasePlan] = {}
        for name, spec in curriculum.items():
            self.phases[name] = PhasePlan(
                name=name,
                target_ratio=float(spec.get("target_ratio", 0.5)),
                general_ratio=float(spec.get("general_ratio", 0.5)),
                tokens=int(spec.get("tokens", 0)),
            )

    def phase(self, name: str) -> PhasePlan:
        if name not in self.phases:
            raise KeyError(f"Unknown curriculum phase '{name}'. Have {list(self.phases)}")
        return self.phases[name]

    def allocate(self, phase_name: str, total_tokens: int) -> dict[str, int]:
        phase = self.phase(phase_name)
        total = phase.ratio_sum or 1.0
        target = int(total_tokens * phase.target_ratio / total)
        general = total_tokens - target
        return {"target": target, "general": general}
