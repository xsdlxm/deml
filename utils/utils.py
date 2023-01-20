#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:26:53 2020

@author: xsdlxm
"""

import json, os
from pathlib import Path
import pandas as pd
import numpy as np


def check_dir_existence(dir, ischeck=True, is_info=False):
    if ischeck:
        path = Path(dir)
        if not os.path.exists(path):
            print("Making dir %s " % path)
            os.mkdir(path)
        else:
            print("dir %s exists" % path)

def get_best_formula(structure):
    structure = structure.get_primitive_structure()
    formula = structure.composition.formula
    # print(formula)
    a = formula.split(' ')
    b = [1, 2, 3]
    for i in a:
        if '1' in i:
            b[1] = i.strip('1')
        if 'B2' in i:
            b[2] = i
        if '2' in i and 'B' not in i:
            b[0] = i
    return b[0] + b[1] + b[2]

def read_txt(file):
    data = []
    with open(file, 'r') as f:
        con = f.readlines()
        for row in con:
            tmp_list = row.split()
            tmp_list[-1] = tmp_list[-1].replace(r'\n', '')
            data.append(tmp_list)
    return data

def read_json(fjson):
    """
    Args:
        fjson (str) - file name of json to read
    
    Returns:
        dictionary stored in fjson
    """
    with open(fjson) as f:
        return json.load(f)

def write_json(d, fjson):
    """
    Args:
        d (dict) - dictionary to write
        fjson (str) - file name of json to write
    
    Returns:
        written dictionary
    """        
    with open(fjson, 'w') as f:
        json.dump(d, f)
    return d

def df_to_excell(path, df):
    if '.xlsx' in path:
        write = pd.ExcelWriter(path)
        df.to_excel(write)
        write.save()

