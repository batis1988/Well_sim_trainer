def test_operating_point_within_bounds():
    """Рабочая точка должна быть между 0 и Pr"""
    # ... setup ...
    point = analyzer.solve()
    assert 0 <= point.pwf <= ipr.pr
    assert point.rate >= 0

def test_thp_increase_reduces_rate():
    """Увеличение устьевого давления снижает дебит"""
    vlp1 = VLPModel(fluid, 8000, 2.875, thp=200)
    vlp2 = VLPModel(fluid, 8000, 2.875, thp=500)
    op1 = NodalAnalyzer(ipr, vlp1).solve()
    op2 = NodalAnalyzer(ipr, vlp2).solve()
    assert op2.rate < op1.rate

def test_no_solution_when_vlp_above_ipr():
    """Если VLP полностью выше IPR — нет решения"""
    vlp_extreme = VLPModel(fluid, 8000, 2.875, thp=3900)  # THP почти как Pr
    point = NodalAnalyzer(ipr, vlp_extreme).solve()
    assert not point.converged