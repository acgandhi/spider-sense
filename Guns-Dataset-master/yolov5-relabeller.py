# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 12:31:10 2021

@author: derph
"""

import os
from PIL import Image

directory = "./Old_Labels/"
for filename in os.listdir(directory):
    os.rename("./Images/" + filename[:-4] + ".jpeg", "./Images/extra-pistols" + filename[:-4] + ".jpeg")
    imagePath = "./Images/extra-pistols" + filename[:-4] + ".jpeg"
    im = Image.open(imagePath)
    width, height = im.size
    im.close()
    print(imagePath, width, height)
    os.rename(imagePath, "./Images/extra-pistols" + filename[:-4] + ".jpeg")
    with open(directory + filename, 'r') as file:
        lines = file.read().split("\n")
        numDet = int(lines[0])
        print(numDet)
        annot = ""
        for i in range(1, numDet + 1):
            coord = [int(num) for num in lines[i].split(" ")]
            x_center = str(round((coord[0] + (coord[2] - coord[0])/2)/width, 2)) + " "
            y_center =  str(round((coord[1] + (coord[3] - coord[1])/2)/height, 2)) + " "
            thisWidth = str(round((coord[2] - coord[0])/width, 2)) + " "
            thisHeight = str(round((coord[3] - coord[1])/height, 2)) + " "
            annot += "0 " + x_center + y_center + thisWidth + thisHeight + "\n"
        with open("./Labels/extra-pistols" + filename, 'w') as newFile:
            newFile.write(annot)
