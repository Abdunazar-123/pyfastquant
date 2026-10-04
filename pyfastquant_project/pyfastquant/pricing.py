"""Result types returned by the Python API."""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict


@dataclass
class PriceResult:
    price: float
    std_error: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    computation_time_ms: Optional[float] = None
    used_antithetic: Optional[bool] = None
    greeks: Optional[Dict[str, float]] = None

    def __str__(self) -> str:
        lines = [f"Price: ${self.price:.4f}"]
        if self.std_error is not None:
            lines.append(f"Std error: +/-${self.std_error:.4f}")
        if self.confidence_interval:
            lines.append(
                f"95% CI: [${self.confidence_interval[0]:.4f}, ${self.confidence_interval[1]:.4f}]"
            )
        if self.computation_time_ms is not None:
            lines.append(f"Time: {self.computation_time_ms:.2f} ms")
        if self.used_antithetic is not None:
            lines.append(f"Antithetic variates: {self.used_antithetic}")
        if self.greeks:
            lines.append("Greeks:")
            for k, v in self.greeks.items():
                lines.append(f"  {k.title()}: {v:.6f}")
        return "\n".join(lines)
