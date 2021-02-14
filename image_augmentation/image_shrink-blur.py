# loading library 
import cv2 
import numpy as np 
import os


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images
folder=r"C:\Users\Thelo\OneDrive\Desktop\spidersense\input"

images = load_images_from_folder(folder)

# Specify the kernel size. 
# The greater the size, the more the motion. 
kernel_size = 30

# Create the vertical kernel. 
kernel_v = np.zeros((kernel_size, kernel_size)) 

# Create a copy of the same for creating the horizontal kernel. 
kernel_h = np.copy(kernel_v) 

# Fill the middle row with ones. 
kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size) 
kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size) 

# Normalize. 
kernel_v /= kernel_size 
kernel_h /= kernel_size 

for x in range(len(images)):

    img = images[x]


    # Apply the vertical kernel. 
    vertical_mb = cv2.filter2D(img, -1, kernel_v) 

    # Apply the horizontal kernel. 
    horizonal_mb = cv2.filter2D(img, -1, kernel_h) 

    temp_name1 = 'blur_vertical' + str(x) + '.jpg'
    temp_name2 = 'blur_horizontal' + str(x) + '.jpg'
    

    # Save the outputs. 
    cv2.imwrite(temp_name1, vertical_mb) 
    cv2.imwrite(temp_name2, horizonal_mb) 

    cv2.imwrite(r'C:\Users\Thelo\OneDrive\Desktop\spidersense\output',output) 

