import numpy as np
from dataclasses import dataclass
from src.well_sim import constants as C
from src.well_sim.config import DEFAULT_CONFIG

@dataclass
class FluidProperties:
    api_gravity: float
    gor: float
    sg_gas: float = 0.65
    temp_reservoir: float = 200.0
    config: object = None  # Инъекция конфига

    def __post_init__(self):
        if self.config is None:
            self.config = DEFAULT_CONFIG["pvt"]

    def bubble_point_pressure(self) -> float:
        cfg = self.config
        y = 0.00091 * self.temp_reservoir - 0.0125 * self.api_gravity
        # Использование констант из конфига вместо 18.2 и 0.83
        return cfg.STANDING_C1 * ((self.gor / self.sg_gas)**cfg.STANDING_GOR_EXP * 10**y - 1.4)

    def oil_fvf(self, p: float) -> float:
        cfg = self.config
        pb = self.bubble_point_pressure()
        
        if p >= pb:
            bob = self._bo_at_pb()
            bo = bob * np.exp(-cfg.OIL_COMPRESSIBILITY_1_PSI * (p - pb))
            return min(max(bo, 0.5), C.MAX_FVF_RB_STB)
        else:
            rs = self.gor * (p / max(pb, 1.0))**cfg.STANDING_GOR_EXP
            bo = 1.0 + cfg.BO_C1 * rs + cfg.BO_C2 * (self.temp_reservoir - cfg.BO_TEMP_REF_F) * rs / self.api_gravity
            return min(max(bo, 0.5), C.MAX_FVF_RB_STB)

    def _bo_at_pb(self) -> float:
        cfg = self.config
        rs = self.gor
        bo = 1.0 + cfg.BO_C1 * rs + cfg.BO_C2 * (self.temp_reservoir - cfg.BO_TEMP_REF_F) * rs / self.api_gravity
        return min(max(bo, 0.5), C.MAX_FVF_RB_STB)

    def oil_viscosity(self, p: float) -> float:
        cfg = self.config
        pb = self.bubble_point_pressure()
        
        x = 10**(3.0324 - 0.02023 * self.api_gravity)
        mu_od = 10**(x * self.temp_reservoir**(-1.163)) - 1.0
        mu_od = max(mu_od, cfg.MU_DEAD_MIN_CP)

        if p >= pb:
            try:
                exponent = -11.513 - 8.98e-5 * p
                p_term = p**1.187
                exp_term = np.exp(exponent)
                if p_term > 1e300 or exp_term > 1e300:
                    m = C.MAX_VISCOSITY_CP
                else:
                    m = 2.6 * p_term * exp_term
                    m = min(m, C.MAX_VISCOSITY_CP)
            except (OverflowError, FloatingPointError):
                m = C.MAX_VISCOSITY_CP
            mu = mu_od * (p / max(pb, 1.0))**m
        else:
            rs = self.gor * (p / max(pb, 1.0))**cfg.STANDING_GOR_EXP
            rs = max(rs, 1.0)
            a = 10.715 * rs**(-0.515)
            b = 5.44 * rs**(-0.338)
            mu = a * mu_od**b

        return min(max(mu, C.MIN_VISCOSITY_CP), C.MAX_VISCOSITY_CP)