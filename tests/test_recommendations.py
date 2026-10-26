from src.ui.recommendations import recommendation_message
def test_idle_copy_is_not_contradictory(): assert recommendation_message(0,0,0,15,3,1,0,1)=="Queue is currently clear. Existing capacity is sufficient."
def test_counter_action_copy(): assert "from 3 to 4" in recommendation_message(20,10,30,15,3,4,2,1)
