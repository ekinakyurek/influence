import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pickle
import logging
import os
import shutil
import copy
import sys
import random, math

def setLogger(logger, LOG_FN):
    logger.handlers = []
    fileHandler = logging.FileHandler(LOG_FN, mode = 'w') #could also be 'a'
    logFormatter = logging.Formatter("%(asctime)s [%(funcName)-15s] [%(levelname)-5.5s]  %(message)s")
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

