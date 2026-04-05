"""Entry point for Flask app."""
import os
from pathlib import Path
from app import create_app

# Load .env file from parent directory
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=8080)