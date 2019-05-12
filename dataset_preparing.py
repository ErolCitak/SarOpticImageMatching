import os
import numpy as np
import h5py
import cv2

"""
    1) Read data w.r.t. weather
    2) Save each folder as a hdf5 file
    
    
    Tip: Folders are designed like below;
        
         * Spring_Sar        * Spring_Optic
            sar_im_1            opt_im_1
            sar_im_2            opt_im_2
            sar_im_3            opt_im_3
            

Author:
Erol Citak
https://www.linkedin.com/in/erolcitak/?locale=en_US

30/April/2019
"""
def read_images_sequentially(path, name=""):
    folders = os.listdir(path)
    number_of_folders = len(folders)

    #hf_sar = h5py.File(name+'_sar.h5', 'w')
    #hf_opt = h5py.File(name+'_opt.h5', 'w')

    # foreach subfolder; open and save mutual images
    sar_images = []
    opt_images = []

    for i in range(int(number_of_folders/2)):
        sar_subfolder_name = folders[i]
        opt_subfolder_name = folders[i + int(number_of_folders / 2)]

        sar_subfolder_path = os.path.join(spring_path,sar_subfolder_name)
        opt_subfolder_path = os.path.join(spring_path,opt_subfolder_name)

        for images_sar in os.listdir(sar_subfolder_path):
            sar_images.append(cv2.imread(os.path.join(sar_subfolder_path,images_sar)))

        for images_opt in os.listdir(opt_subfolder_path):
            opt_images.append(cv2.imread(os.path.join(opt_subfolder_path,images_opt)))

    print("SAR SHAPE: ",np.array(sar_images).shape)
    print("OPT SHAPE: ", np.array(opt_images).shape)

if __name__=="__main__":
    print("Optic-Sar Matching dataset preparation!...")

    # Images - Path definition
    spring_path = "D://Sen1_2//Spring"
    summer_path = "D://Sen1_2//Summer"
    fall_path = "D://Sen1_2//Fall"
    winter_path = "D://Sen1_2//Winter"

    read_images_sequentially(spring_path,"erol")
