from pathlib import Path
import yaml


BASE_DIR = Path(__file__).resolve().parent.parent
LOCATIONS_FILE = BASE_DIR / "config" / "locations.yaml"


def load_locations():
    with open(LOCATIONS_FILE, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if "locations" not in data:
        raise ValueError("The locations.yaml file does not contain a 'locations' section.")

    return data["locations"]


def get_enabled_locations():
    locations = load_locations()

    return [
        location
        for location in locations
        if location.get("enabled", False)
    ]


def get_location_by_id(location_id):
    locations = load_locations()

    for location in locations:
        if location["id"] == location_id:
            return location

    raise ValueError(f"Location with id '{location_id}' was not found.")