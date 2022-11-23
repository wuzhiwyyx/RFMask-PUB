'''
 # @ Author: Zhi Wu
 # @ Create Time: 2022-07-25 00:45:25
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-07-25 00:46:29
 # @ Description: Logger definition.
 '''


import datetime
import logging
import os
import sys
import time
from logging import Handler, StreamHandler, getLevelName


def build_logger(config, phase='train', model_name='Model'):
    log_dir = 'log'
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger_ddp(config.exper, log_dir, time.strftime('%Y-%m-%d-%H-%M'), 0, model_name, phase)
    return logger

def get_logger(name, save_dir, verbosity=1, filename="log.txt"):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(os.path.join(save_dir, filename), "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger

# this class is a copy of logging.FileHandler except we end self.close()
# at the end of each emit. While closing file and reopening file after each
# write is not efficient, it allows us to see partial logs when writing to
# fused Azure blobs, which is very convenient
class FileHandler(StreamHandler):
    """
    A handler class which writes formatted logging records to disk files.
    """
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        """
        Open the specified file and use it as the stream for logging.
        """
        # Issue #27493: add support for Path objects to be passed in
        filename = os.fspath(filename)
        #keep the absolute path, otherwise derived classes which use this
        #may come a cropper when the current directory changes
        self.baseFilename = os.path.abspath(filename)
        self.mode = mode
        self.encoding = encoding
        self.delay = delay
        if delay:
            #We don't open the stream, but we still need to call the
            #Handler constructor to set level, formatter, lock etc.
            Handler.__init__(self)
            self.stream = None
        else:
            StreamHandler.__init__(self, self._open())

    def close(self):
        """
        Closes the stream.
        """
        self.acquire()
        try:
            try:
                if self.stream:
                    try:
                        self.flush()
                    finally:
                        stream = self.stream
                        self.stream = None
                        if hasattr(stream, "close"):
                            stream.close()
            finally:
                # Issue #19523: call unconditionally to
                # prevent a handler leak when delay is set
                StreamHandler.close(self)
        finally:
            self.release()

    def _open(self):
        """
        Open the current base file with the (original) mode and encoding.
        Return the resulting stream.
        """
        return open(self.baseFilename, self.mode, encoding=self.encoding)

    def emit(self, record):
        """
        Emit a record.

        If the stream was not opened because 'delay' was specified in the
        constructor, open it before calling the superclass's emit.
        """
        if self.stream is None:
            self.stream = self._open()
        StreamHandler.emit(self, record)
        self.close()

    def __repr__(self):
        level = getLevelName(self.level)
        return '<%s %s (%s)>' % (self.__class__.__name__, self.baseFilename, level)

def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()

def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logging.Formatter.converter = beijing
    logger.addHandler(ch)

    if save_dir:
        fh = FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def setup_logger_ddp(exp_name, save_dir, time_str, distributed_rank, model_name, phase='train'):
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logging.Formatter.converter = beijing
    logger.addHandler(ch)

    if save_dir:
        
        log_file = '{}_{}_{}.log'.format(exp_name, time_str, phase)
        fh = FileHandler(os.path.join(save_dir, log_file))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logging.Formatter.converter = beijing
        logger.addHandler(fh)

    return logger