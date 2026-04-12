"""File watcher — monitors OOTP export directory for new/changed CSVs."""
import time
import logging
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from app.core.database import load_config, init_db
from app.core.ingestion import ingest_file
from app.core.recommendations import generate_recommendations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class OOTPFileHandler(FileSystemEventHandler):
    """Handle CSV file creation/modification events."""

    def __init__(self, debounce_seconds=2):
        self.debounce_seconds = debounce_seconds
        self._pending = {}

    def on_modified(self, event):
        self._handle(event)

    def on_created(self, event):
        self._handle(event)

    def _handle(self, event):
        if event.is_directory:
            return
        if not event.src_path.lower().endswith('.csv'):
            return

        # Debounce: record timestamp, process after delay
        self._pending[event.src_path] = time.time()

    def process_pending(self):
        """Check and process any pending files past the debounce window."""
        now = time.time()
        to_process = []
        for path, timestamp in list(self._pending.items()):
            if now - timestamp >= self.debounce_seconds:
                to_process.append(path)
                del self._pending[path]

        for path in to_process:
            try:
                logger.info(f"Processing: {Path(path).name}")
                result = ingest_file(path)
                logger.info(f"  Result: {result}")
                if result.get('status') == 'success':
                    logger.info("  Regenerating recommendations...")
                    generate_recommendations()
                    logger.info("  Done.")
            except Exception as e:
                logger.error(f"  Error processing {path}: {e}")


def start_watcher():
    """Start the file watcher."""
    config = load_config()
    watch_dir = config.get('watch_directory', '')
    debounce = config.get('watcher', {}).get('debounce_seconds', 2)
    poll_interval = config.get('watcher', {}).get('poll_interval_seconds', 5)

    if not watch_dir or not Path(watch_dir).exists():
        logger.error(f"Watch directory not found: {watch_dir}")
        return

    # Ensure DB is initialized
    init_db()

    handler = OOTPFileHandler(debounce_seconds=debounce)
    observer = Observer()
    observer.schedule(handler, watch_dir, recursive=False)
    observer.start()

    logger.info(f"Watching: {watch_dir}")
    logger.info("Press Ctrl+C to stop")

    try:
        while True:
            handler.process_pending()
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == '__main__':
    start_watcher()
