import numpy as np
from src.well_sim import constants as C
from src.well_sim.config import DEFAULT_CONFIG


class ChokeModel:
    """Модель поверхностного штуцера (критический + докритический режим).
    
    Критический режим: downstream-давление не влияет на дебит (скважина
    "не слышит" сепаратор). Докритический: инверсия диафрагменной модели.
    """

    def __init__(self, size_64: float, correlation: str = None, config=None):
        """
        Args:
            size_64: Диаметр штуцера в 1/64 дюйма (bean size: 32 = 0.5")
        """
        self.config = config if config is not None else DEFAULT_CONFIG["choke"]
        self.size_64 = size_64
        self.correlation = correlation or self.config.CORRELATION
        
        coef = C.CHOKE_CORRELATIONS[self.correlation]
        self.A = coef["A"]
        self.B = coef["B"]
        self.Cexp = coef["C"]

    def p_critical_upstream(self, q_liq: float, gor: float) -> float:
        """Устьевое давление в критическом режиме, psia"""
        if q_liq <= 0:
            return 0.0
        gor_eff = max(gor, self.config.MIN_GOR_SCF_STB)
        return self.A * q_liq * gor_eff**self.B / self.size_64**self.Cexp

    def p_upstream(self, q_liq: float, gor: float, p_downstream: float) -> float:
        """Требуемое устьевое давление для пропуска q_liq через штуцер.
        
        Критический режим: P_up = Gilbert(Q).
        Докритический: инверсия диафрагмы 
        P_up = sqrt(P_up_c² - P_down_c² + P_down²) — непрерывно на границе.
        """
        p_up_c = self.p_critical_upstream(q_liq, gor)
        p_down_c = self.config.CRITICAL_RATIO * p_up_c

        if p_downstream <= p_down_c:
            return p_up_c  # Критический поток: сепаратор не влияет

        val = p_up_c**2 - p_down_c**2 + p_downstream**2
        return max(np.sqrt(max(val, 0.0)), p_downstream)