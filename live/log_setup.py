import logging
from datetime import datetime

class CustomFormatter(logging.Formatter):
    def format(self, record):
        # TODO: Add warning logs and above to DB
        log_lvl = record.levelname
        msg = record.getMessage()
        return f"{datetime.now()} - {log_lvl} - {msg}"

def setup_logger(logging_lvl):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging_lvl)

    ch = logging.StreamHandler()
    ch.setLevel(logging_lvl)

    formatter = CustomFormatter()

    ch.setFormatter(formatter)
    
    logger.addHandler(ch)

    return logger