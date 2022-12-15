# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 16:55:28 2021

@author: OK
"""
import argparse
def str2bool(v):
    """
    Transform user input(argument) to be boolean expression.
    :param v: (string) user input
    :return: Bool(True, False)
    """
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
