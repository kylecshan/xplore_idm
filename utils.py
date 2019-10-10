# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:08:28 2019

@author: kcsii_000
"""

import itertools

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))