import numpy as np
from src.well_sim.core.pvt import FluidProperties
from src.well_sim import constants as C
from src.well_sim.config import DEFAULT_CONFIG

class VLPModel:
    """
    Упрощенная VLP для вертикальной двухфазной скважины.
    Версия 3.0: Исправлена форма кривой (U-shape), интегрированы конфиги.
    """
    
    def __init__(self, fluid: FluidProperties, depth: float,
                 tubing_id: float, thp: float, config=None):
        self.fluid = fluid
        self.depth = depth
        self.tubing_id = tubing_id
        self.thp = thp
        # Используем конфиг из переданного аргумента или дефолтный
        self.config = config if config is not None else DEFAULT_CONFIG["outflow"]
        
        # Геометрия с использованием констант
        self.d_ft = tubing_id * C.IN_TO_FT
        self.area_ft2 = C.PI * self.d_ft**2 / 4.0

    def pwf(self, q_oil: float, wor: float = 0.0, gor_current: float = None) -> float:
        cfg = self.config
        
        # 1. Статический режим
        if q_oil <= 1e-6:
            return self._static_pwf()
        
        gor = gor_current if gor_current is not None else self.fluid.gor
        
        # Среднее давление для PVT (упрощение для скорости)
        p_avg = self.thp + self.depth * 0.15
        bo = max(self.fluid.oil_fvf(max(p_avg, 100.0)), 0.5)

        # 2. Скорости (Superficial velocities)
        vol_liq = q_oil * bo * C.BBL_TO_FT3 / C.DAY_TO_SEC
        v_sl = vol_liq / self.area_ft2

        t_avg_r = 460.0 + self.fluid.temp_reservoir - 40.0
        # Bg calculation using standard conditions from constants
        bg = 0.0283 * t_avg_r / max(p_avg, C.P_STD_PSIA)
        
        vol_gas = q_oil * gor * bg / C.DAY_TO_SEC
        v_sg = vol_gas / self.area_ft2
        v_m = v_sl + v_sg

        # 3. Входные параметры смеси
        vg_std = gor / C.BBL_TO_FT3
        vl_std = bo
        lambda_l = vl_std / (vl_std + vg_std)
        lambda_l = max(min(lambda_l, 0.99), 0.001)

        # Число Фруда
        fr = v_m / max(np.sqrt(C.G_FT_S2 * self.d_ft), 0.01)
        
        # 4. Плотности
        rho_oil = C.OIL_DENSITY_REF_LB_FT3 / bo
        rho_gas = C.GAS_DENSITY_REF_LB_FT3 * self.fluid.sg_gas / max(bg, 0.001)
        
        # === 5. HOLDUP (Непрерывная модель) ===
        # А. Статический предел (Q -> 0)
        pwf_static = self._static_pwf()
        target_dp_hydro_at_zero = pwf_static - self.thp
        rho_mix_target = target_dp_hydro_at_zero * C.PSI_TO_LBF_FT2 / self.depth
        
        if abs(rho_oil - rho_gas) > 1e-6:
            hl_zero = (rho_mix_target - rho_gas) / (rho_oil - rho_gas)
        else:
            hl_zero = 0.5
        hl_zero = max(min(hl_zero, 0.98), 0.05)
        
        # Б. Потоковый предел (Q -> inf)
        # Не позволяем Hl падать ниже определенного предела, даже при высоких скоростях
        # Это имитирует наличие жидкой пленки на стенках труб
        hl_floor = 0.05  # Минимум 5% жидкости всегда остается
        hl_inf = max(lambda_l * cfg.HOLDUP_FLOW_FACTOR, hl_floor) 
        hl_inf = max(min(hl_inf, 0.98), 0.05)
        
        # В. Экспоненциальный переход
        # tau берется из конфига. Чем меньше tau, тем быстрее переход к потоку.
        hl = hl_inf + (hl_zero - hl_inf) * np.exp(-fr / cfg.HOLDUP_TAU)
        
        # Г. Физические ограничения
        hl = max(min(hl, 0.98), 0.01)

        # 6. Гидравлика
        rho_mix = hl * rho_oil + (1.0 - hl) * rho_gas
        dp_hydro = rho_mix / C.PSI_TO_LBF_FT2 * self.depth

        # Трение
        mu = max(self.fluid.oil_viscosity(max(p_avg, 100.0)), C.MIN_VISCOSITY_CP)
        
        # Используем плотность НЕТТО (без учета holdup) для расчета Re, 
        # чтобы избежать искусственного занижения трения при низком Hl
        # Или используем rho_mix, но ограничиваем его снизу
        rho_mix_friction = max(rho_mix, rho_gas * 5.0) # Плотность не ниже 5-кратной газовой
        
        re = 1488.0 * rho_mix_friction * v_m * self.d_ft / max(mu, 0.01)
        re = max(re, 10.0)

        if re < 2000:
            f = cfg.FRICTION_LAMINAR_CONST / re
        else:
            eps_rel = C.ROUGHNESS_STEEL_FT / self.d_ft
            arg = eps_rel / 3.7 + 5.74 / re**0.9
            denom = np.log10(max(arg, 1e-10))
            f = 0.25 / abs(denom)**2
            # Clamp f
            f = max(min(f, cfg.FRICTION_TURBULENT_MAX), cfg.FRICTION_TURBULENT_MIN)

        # Дарси-Вейсбах
        dp_fric = f * (self.depth / self.d_ft) * rho_mix_friction * v_m**2 / (2.0 * C.G_FT_S2 * C.PSI_TO_LBF_FT2)
        
        # === ДОПОЛНИТЕЛЬНАЯ КИНЕТИЧЕСКАЯ ПОПРАВКА (Ускорение) ===
        # При высоких скоростях газа значимую роль играет ускорение смеси
        # dp_acc = rho_mix * v_m^2 / (144 * g) -- упрощенно
        # Это гарантирует рост давления при очень больших Q
        dp_acc = rho_mix * v_m**2 / (C.PSI_TO_LBF_FT2 * C.G_FT_S2) 

        pwf_calc = self.thp + dp_hydro + dp_fric + dp_acc
        
        # === 7. ГАРАНТИЯ ФИЗИЧЕСКОЙ КОРРЕКТНОСТИ ===
        # Мы НЕ ограничиваем давление сверху жестким клампом (как было раньше),
        # чтобы позволить трению поднять Pwf на высоких дебитах (формирование U-кривой).
        # Единственное ограничение: на старте (малые Q) давление не должно превышать статику.
        
        # if pwf_calc > pwf_static:
        #     # Если расчетное давление выше статического (численный шум или артефакт старта),
        #     # прижимаем к статике. Это гарантирует монотонный старт.
        #     pwf_calc = pwf_static
            
        return max(min(pwf_calc, C.MAX_PRESSURE_PSIA), self.thp)

    def _static_pwf(self) -> float:
        """Статическое Pwf с использованием конфига"""
        cfg = self.config
        p_avg = self.thp + 1000.0
        bo = max(self.fluid.oil_fvf(max(p_avg, 100.0)), 0.5)
        rho_oil = C.OIL_DENSITY_REF_LB_FT3 / bo
        
        # Используем параметр затухания из конфига вместо хардкода 350.0
        gas_factor = np.exp(-self.fluid.gor / cfg.STATIC_GOR_DECAY)
        eff_grad = rho_oil / C.PSI_TO_LBF_FT2 * gas_factor
        
        pwf_static = self.thp + eff_grad * self.depth
        return max(min(pwf_static, C.MAX_PRESSURE_PSIA), self.thp)