
import logging

def setup_logger(level=logging.INFO):
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    return logger
