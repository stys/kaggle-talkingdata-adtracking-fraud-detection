# -*- coding: utf-8 -*-
from os import makedirs as os_makedirs
import errno


def makedirs(path):
    try:
        os_makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise