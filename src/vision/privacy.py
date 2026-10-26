"""Privacy defaults for local, anonymous queue analytics."""
from pathlib import Path
import yaml
def load_privacy_config():
 path=Path(__file__).resolve().parents[2]/"config"/"privacy.yaml"
 return yaml.safe_load(path.read_text(encoding="utf-8"))
