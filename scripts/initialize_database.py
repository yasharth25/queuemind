from src.config import load_locations
from src.storage.repository import QueueRepository


if __name__ == "__main__":
    repository = QueueRepository()
    repository.seed_locations(load_locations())
    print("Initialized QueueMind SQLite database and seeded locations.")
