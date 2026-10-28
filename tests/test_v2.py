from src.ui.v2 import alerts, location_state
from src.config import get_enabled_locations
def test_command_center_location_state_and_alerts_are_backend_derived():
 locations=get_enabled_locations(); state=location_state(locations[0]); assert state["queue20"]>=0 and state["recommendation"]["recommended_counters"]>=1
 assert isinstance(alerts(locations),list)
