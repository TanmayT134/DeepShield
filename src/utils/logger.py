import logging
import os

LOG_FOLDER = "outputs/logs"

os.makedirs(LOG_FOLDER, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(LOG_FOLDER, "deepshield.log")
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("DeepShield")