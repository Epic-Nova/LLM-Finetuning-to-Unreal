import logging
import time
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


class TqdmLoggingHandler(logging.Handler):
    """Redirect logging messages through tqdm.write() so the progress bar stays clean."""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


# Setup logger
logger = logging.getLogger("tqdm_logger")
logger.setLevel(logging.DEBUG)

handler = TqdmLoggingHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Optional: mute other logging output
logger.propagate = False


def main():
    logger.info("🚀 Starting logging spam with progress bar...")
    total = 1000
    with tqdm(total=total, desc="Logging loop") as pbar:
        for i in range(total):
            if i % 100 == 0:
                logger.warning(f"⚠️ Warning at step {i}")
            elif i % 200 == 0:
                logger.error(f"🔥 Error at step {i}")
            else:
                logger.info(f"🔄 Log entry {i}")
            time.sleep(0.005)
            pbar.update(1)
    logger.info("✅ Done with logging and progress!")


if __name__ == "__main__":
    with logging_redirect_tqdm():
        main()