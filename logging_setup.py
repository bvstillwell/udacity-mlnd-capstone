# #############################################################################
#
# This file contains he loggingsetup
#
import os
import datetime
import logging


def get_datetime_filename():
    """Return a filename that contains a timestamp"""
    return datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')

#
# Setup logging
#
if not os.path.exists('log'):
    os.makedirs('log')
log = logging.getLogger('')
session_id = get_datetime_filename()
logfile = os.path.join('log', session_id + '.log')
logging.basicConfig(filename=logfile, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())


def log(message):
    logging.info(message)
