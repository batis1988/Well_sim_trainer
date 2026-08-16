"""
Конфигурация моделей и калибровочные коэффициенты.
Эти параметры МОЖНО и НУЖНО менять для настройки точности симулятора.
"""

from dataclasses import dataclass, field

@dataclass
class PVTConfig:
    """Коэффициенты для PVT корреляций (Standing, Beggs-Robinson)"""
    # Standing Pb correlation
    STANDING_C1: float = 18.2
    STANDING_GOR_EXP: float = 0.83
    
    # Oil Compressibility (Undersaturated)
    OIL_COMPRESSIBILITY_1_PSI: float = 5e-5  # co
    
    # Standing Bo correlation coeffs
    BO_C1: float = 4.67e-4
    BO_C2: float = 1.75e-5
    BO_TEMP_REF_F: float = 60.0
    
    # Viscosity limits
    MU_DEAD_MIN_CP: float = 0.01
    
@dataclass
class InflowConfig:
    """Коэффициенты IPR (Vogel)"""
    # Vogel equation: Q = Qmax * (1 - A*(Pwf/Pr) - B*(Pwf/Pr)^2)
    VOGEL_A: float = 0.2
    VOGEL_B: float = 0.8
    VOGEL_DIVISOR: float = 1.8  # Делитель для Qmax теоретического

@dataclass
class OutflowConfig:
    """Коэффициенты гидравлики ствола (VLP)"""
    # Friction factor
    FRICTION_LAMINAR_CONST: float = 64.0  # f = 64/Re
    FRICTION_TURBULENT_MIN: float = 0.005
    FRICTION_TURBULENT_MAX: float = 0.1
    
    # Holdup Correlation (Custom Exponential)
    # Hl = Hl_flow + (Hl_static - Hl_flow) * exp(-Fr / tau)
    HOLDUP_TAU: float = 2.0          # Константа затухания перехода (калибровка!)
    HOLDUP_FLOW_FACTOR: float = 1.2  # Множитель для no-slip holdup (slip effect)
    
    # Static Pressure Calibration
    # Pwf_static gradient reduction based on GOR
    STATIC_GOR_DECAY: float = 200.0  # Чем меньше, тем сильнее газ облегчает статику
    
    # Solver Limits
    SOLVER_PWF_MIN_PSIA: float = 100.0
    SOLVER_Q_TOLERANCE: float = 1.0

# Глобальный инстанс конфигурации по умолчанию
DEFAULT_CONFIG = {
    "pvt": PVTConfig(),
    "inflow": InflowConfig(),
    "outflow": OutflowConfig()
}