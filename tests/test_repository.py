from src.storage.repository import QueueRepository
from src.synthetic.generator import generate_synthetic_queue_data


LOCATIONS = [{"id": "bank", "display_name": "Bank", "category": "bank", "default_counters": 2, "average_service_time_minutes": 3, "target_wait_minutes": 10, "warning_queue_length": 5, "critical_queue_length": 10, "enabled": True}]


def test_repository_seeds_and_imports_synthetic_observations():
    repository = QueueRepository(":memory:")
    repository.seed_locations(LOCATIONS)
    data = generate_synthetic_queue_data(LOCATIONS, "2026-01-01", days=1)

    assert repository.import_observations(data) == len(data)
    recent = repository.get_recent_observations("bank", 5)
    assert len(recent) == 5
    assert set(recent["data_source"]) == {"SYNTHETIC"}
