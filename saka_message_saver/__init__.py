import pathlib
import datetime
from logging import getLogger, FileHandler, StreamHandler, Formatter, DEBUG, INFO, WARNING, ERROR, CRITICAL


PROJECT_ROOT_PATH = pathlib.Path(__file__).parent.parent
EXEC_DATE_STR = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# logger
logger = getLogger(__name__)
logger.setLevel(DEBUG)

formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# stream handler
stream_handler = StreamHandler()
stream_handler.setLevel(DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

from .saka_message_saver import *
from .parameter.parameters import *
from .setting_ui import *
