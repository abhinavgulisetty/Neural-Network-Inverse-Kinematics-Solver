"""
Start the Flask web dashboard.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from web.app import app

if __name__ == "__main__":
    print("=" * 60)
    print("Neural IK Solver — Web Dashboard")
    print("=" * 60)
    print("Open http://localhost:5000 in your browser\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
