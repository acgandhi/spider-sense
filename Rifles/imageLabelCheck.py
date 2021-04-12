# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 18:25:45 2021

@author: derph
"""

import os

missingIm = []
for label in os.scandir("./Labels/"):
    label = label.name
    image = "./Images/" + label[:-4] + ".jpg"
    if os.path.exists(image) is False:
        missingIm.append(image)
print(missingIm)
missingLab = []
for image in os.scandir("./Images/"):
    image = image.name
    label = "./Labels/" + image[:-4] + ".txt"
    if os.path.exists(label) is False:
        missingLab.append(label)
print(missingLab)