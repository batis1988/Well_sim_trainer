import numpy as np
from scipy.optimize import brentq
from .inflow import IPRModel
from .outflow import VLPModel
from dataclasses import dataclass

@dataclass
class OperatingPoint:
    rate: float      # stb/day
    pwf: float       # psia
    converged: bool
    
class NodalAnalyzer:
    def __init__(self, ipr: IPRModel, vlp: VLPModel):
        self.ipr = ipr
        self.vlp = vlp
        
    def solve(self, tol: float = 1.0, q_max_reasonable: float = None, 
          pwf_min: float = 100.0) -> OperatingPoint:
        """Найти рабочую точку"""
        if q_max_reasonable is None:
            q_max_reasonable = self.ipr.rate(0) * 1.2
        
        def objective(pwf):
            # Запрещаем решения с Pwf < pwf_min (физический предел)
            if pwf < pwf_min:
                return pwf - pwf_min  # Штраф
            
            q_ipr = self.ipr.rate(pwf)
            if q_ipr > q_max_reasonable:
                return pwf - self.vlp.pwf(q_max_reasonable)
            if q_ipr <= 0:
                return pwf - self.vlp.pwf(0)
            
            q_vlp_pwf = self.vlp.pwf(q_ipr)
            return pwf - q_vlp_pwf
        
        # Поиск в диапазоне [pwf_min, Pr]
        try:
            pwf_sol = brentq(objective, pwf_min, self.ipr.pr, xtol=tol)
            q_sol = self.ipr.rate(pwf_sol)
            return OperatingPoint(rate=q_sol, pwf=pwf_sol, converged=True)
        except ValueError:
            # Fallback на сетку
            pwf_grid = np.linspace(pwf_min, self.ipr.pr, 200)
            obj_grid = np.array([objective(p) for p in pwf_grid])
            idx_min = np.argmin(np.abs(obj_grid))
            pwf_sol = pwf_grid[idx_min]
            q_sol = self.ipr.rate(pwf_sol)
            return OperatingPoint(rate=q_sol, pwf=pwf_sol, converged=False)

    def generate_curves(self, n_points: int = 500):
        """Генерация точек для визуализации"""
        # Единая сетка по Pwf
        pwf_range = np.linspace(0, self.ipr.pr, n_points)
        
        # IPR: прямая зависимость
        q_ipr = np.array([self.ipr.rate(p) for p in pwf_range])
        
        # VLP: обратная зависимость (для каждого Pwf находим Q через бисекцию)
        q_vlp = np.zeros(n_points)
        for i, pwf_target in enumerate(pwf_range):
            def vlp_residual(q):
                return self.vlp.pwf(q) - pwf_target
            
            # Ищем Q в разумном диапазоне
            try:
                from scipy.optimize import brentq
                q_sol = brentq(vlp_residual, 0, self.ipr.rate(0) * 1.5, xtol=1.0)
                q_vlp[i] = q_sol
            except ValueError:
                q_vlp[i] = np.nan  # Нет решения при этом Pwf
        
        # Удаляем NaN для гладкости
        valid = ~np.isnan(q_vlp)
        
        return {
            'ipr': {'pwf': pwf_range[valid], 'rate': q_ipr[valid]},
            'vlp': {'pwf': pwf_range[valid], 'rate': q_vlp[valid]}
        }