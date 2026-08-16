import numpy as np
from src.well_sim.core.pvt import FluidProperties
from src.well_sim.config import DEFAULT_CONFIG

class IPRModel:
    def __init__(self, fluid: FluidProperties, pr: float, pi: float, config=None):
        self.fluid = fluid
        self.pr = pr
        self.pi = pi
        self.pb = fluid.bubble_point_pressure()
        self.config = config or DEFAULT_CONFIG["inflow"]

    def rate(self, pwf: float) -> float:
        cfg = self.config
        pwf = max(pwf, 0.0)
        
        if pwf >= self.pb:
            return self.pi * (self.pr - pwf)
        else:
            q_b = self.pi * (self.pr - self.pb)
            # Использование VOGEL_DIVISOR (1.8) из конфига
            q_max_additional = self.pi * self.pb / cfg.VOGEL_DIVISOR
            ratio = pwf / self.pb
            # Использование коэффициентов A и B из конфига
            q_vogel = q_max_additional * (1 - cfg.VOGEL_A * ratio - cfg.VOGEL_B * ratio**2)
            return q_b + q_vogel
    
    def pressure(self, q: float) -> float:
        """Обратная функция: Pwf по дебиту (для построения кривой)"""
        # Численное решение через бисекцию (надежнее аналитики для composite)
        from scipy.optimize import brentq
        try:
            return brentq(lambda pwf: self.rate(pwf) - q, 0, self.pr)
        except ValueError:
            return 0.0