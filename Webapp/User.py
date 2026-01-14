from pathlib import Path
import json

# Project root = directory containing THIS file
BASE_DIR = Path(__file__).resolve().parent

# Go up one level, then into static/data/user.json
DATA_FILE = BASE_DIR.parent / "static" / "data" / "user.json"


def update_user_json(new_data: dict) -> None:
    # Ensure directories exist
    # Load existing data if possible
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    if DATA_FILE.exists():
        try:
            print('Have file')
            with DATA_FILE.open("r", encoding="utf-8") as f:
                user_data = json.load(f)
                if not isinstance(user_data, dict):
                    user_data = {}
        except json.JSONDecodeError:
            print('Error in file')
            user_data = {}
    else:
        print('No file')
        user_data = {}

    # Merge updates
    user_data.update(new_data)

    # Write file (create or overwrite)
    with DATA_FILE.open("w", encoding="utf-8") as f:
        json.dump(user_data, f, indent=4, ensure_ascii=False)
