import sys
from pathlib import Path
import json

repo_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(repo_root))

from app.monitoring.governance import run_governance_checks

# Load a sample report JSON (create this for testing)
with open('tests/test_governance.json', 'r') as f:
    report = json.load(f)

alerts = run_governance_checks(report, model_version="v1")
print("Governance alerts:", alerts)
