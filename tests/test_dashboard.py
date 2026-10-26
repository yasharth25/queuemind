from src.queueing.simulator import recommend_counter_count


def test_dashboard_backend_recommendation_is_simulation_based():
    result = recommend_counter_count(30, 1.5, 2.5, 3, 12, 20, [2, 3, 4, 5, 6])
    assert result["recommended_outcome"] in result["evaluations"]
