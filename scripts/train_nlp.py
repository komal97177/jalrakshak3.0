import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.nlp import train_all
print(json.dumps(train_all(), indent=1, ensure_ascii=False))
