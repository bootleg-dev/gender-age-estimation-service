import logging

class Logger:

    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s:%(levelname)s:%(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def error(self, msg, *args, **kwargs)-> None:
        if self.logger is not None:
            self.logger.error(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs)-> None:
        if self.logger is not None:
            self.logger.info(msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs)-> None:
        if self.logger is not None:
            self.logger.warning(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs)-> None:
        if self.logger is not None:
            self.logger.warning(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs)-> None:
        if self.logger is not None:
            self.logger.exception(msg, *args, exc_info=True, **kwargs)
