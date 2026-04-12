"""First-run setup script — creates database and validates configuration."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.database import init_db, load_config


def main():
    print("OOTP Perfect Team Optimizer — Setup")
    print("=" * 40)

    # Load and validate config
    try:
        config = load_config()
        print(f"Config loaded: team = {config.get('team_name', 'Unknown')}")
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Check watch directory
    watch_dir = config.get('watch_directory', '')
    if Path(watch_dir).exists():
        print(f"Watch directory OK: {watch_dir}")
        csv_count = len(list(Path(watch_dir).glob('*.csv')))
        print(f"  Found {csv_count} CSV files")
    else:
        print(f"WARNING: Watch directory not found: {watch_dir}")
        print("  You can still import files manually via the dashboard")

    # Initialize database
    print("\nInitializing database...")
    init_db()
    print("Database ready.")

    print("\nSetup complete! Run 'run.bat' to start the optimizer.")


if __name__ == '__main__':
    main()
