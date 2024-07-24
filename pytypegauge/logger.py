import logging
import time
from pathlib import Path

from rich.logging import RichHandler

logger = logging.getLogger(__name__)

# the handler determines where the logs go: stdout/file
shell_handler = RichHandler()

logger.setLevel(logging.DEBUG)
shell_handler.setLevel(logging.DEBUG)

# the formatter determines what our logs will look like
# fmt_shell = "%(levelname)s %(asctime)s %(message)s" # no need level with rich
fmt_shell = "%(asctime)s %(message)s"

shell_formatter = logging.Formatter(fmt_shell)

# here we hook everything together
shell_handler.setFormatter(shell_formatter)

logger.addHandler(shell_handler)
