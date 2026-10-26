from src.demo.scenarios import SCENARIOS, calculate_demo
def test_demo_scenarios_are_synthetic_simulation_backed():
 assert len(SCENARIOS)==5
 result=calculate_demo("Bank Rush",3,1.5)
 assert set(result["projections"])=={5,10,20,30} and result["projections"][20]>=0
