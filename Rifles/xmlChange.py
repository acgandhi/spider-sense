# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 16:22:46 2021

@author: derph
"""

import xmltodict
import os
import json

for annot in os.scandir("./Labels/"):
    annot = annot.path
    print(annot)
    if annot[len(annot)-4:] == ".txt":
        continue
    with open(annot, "r") as a:
        data = xmltodict.parse(a.read())
        data = json.loads(json.dumps(data))["annotation"]
        print(data)
        width = int(data["size"]["width"])
        height = int(data["size"]["height"])
        with open("./Labels/" + data["filename"][:-4] + ".txt", "w") as newAnnot:
            if type(data["object"]) == list:
                for box in data["object"]:
                    name = box["name"]
                    box = box["bndbox"]
                    if name == "pistol":
                        line = "0 "
                    elif name == "rifles":
                        line = "2 "
                    elif name == "knife":
                        line = "1 "
                    line += str((int(box["xmin"]) + (int(box["xmax"]) - int(box["xmin"]))/2)/width) + " "
                    line += str((int(box["ymin"]) + (int(box["ymax"]) - int(box["ymin"]))/2)/height) + " "
                    line += str((int(box["xmax"]) - int(box["xmin"]))/width) + " "
                    line += str((int(box["ymax"]) - int(box["ymin"]))/height) + "\n"
                    newAnnot.write(line)
            else: 
                box = data["object"]["bndbox"]
                name = data["object"]["name"]
                if name == "pistol":
                    line = "0 "
                elif name == "rifles":
                    line = "2 "
                elif name == "knife":
                    line = "1 "
                line += str((int(box["xmin"]) + (int(box["xmax"]) - int(box["xmin"]))/2)/width) + " "
                line += str((int(box["ymin"]) + (int(box["ymax"]) - int(box["ymin"]))/2)/height) + " "
                line += str((int(box["xmax"]) - int(box["xmin"]))/width) + " "
                line += str((int(box["ymax"]) - int(box["ymin"]))/height) + "\n"
                newAnnot.write(line)