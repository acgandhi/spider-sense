# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 20:30:28 2021

@author: derph
"""

import os

directory = "./images"
for filename in os.listdir(directory):
    with open("./Labels/" + filename[:-4] + ".txt", 'w') as file:
        pass