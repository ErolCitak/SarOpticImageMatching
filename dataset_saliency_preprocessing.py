import os
import numpy as np
import cv2
from skimage.measure import compare_ssim
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.measure import compare_psnr

"""

    1) Convert sar image into COLORMAP_HSV
    2) Apply bilateral and average blurring onto sar and optic images with
        different scales
    3) Compute saliency images and save into another folder
    
"""

if __name__ == "__main__":

    # path of images
    sar_images_path = "D://Sen1_2_//SAR"
    optic_images_path = "D://Sen1_2_//OPTIC"

    # saving_paths
    sar_saving_path = "D://Sen1_2_//SAR_SAL"
    optic_saving_path = "D://Sen1_2_//OPTIC_SAL"

    # list images
    sar_images = os.listdir(sar_images_path)
    optic_images = os.listdir(optic_images_path)

    # some needed preparations
    kernel = np.ones((9, 9), np.float32) / 81
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

    scores = []
    sar_sal = []
    opt_sal = []

    seperator = "_"

    for i in range(len(sar_images)):
        sar_image = cv2.imread(os.path.join(sar_images_path, sar_images[i]))

        optic_image_name = sar_images[i].split("_")
        optic_image_name[2] = "s2"
        fully_name = seperator.join(optic_image_name)
        optic_image = cv2.imread(os.path.join(optic_images_path, fully_name))

        ###############################################################################
        ###############################################################################

        # convert it into COLORMAP_HSV

        blured_sar = cv2.bilateralFilter(sar_image, 13, 75, 75)
        colored_sar = cv2.applyColorMap(blured_sar, cv2.COLORMAP_HSV)
        h_s, s_s, v_s = cv2.split(colored_sar)
        blured_sar = cv2.filter2D(h_s, -1, kernel)
        blured_sar = cv2.medianBlur(blured_sar, 11, 2)


        # convert bgr image into hsv and split it into channels
        blured_opt = cv2.bilateralFilter(optic_image, 3, 5, 5)
        hsv_opt = cv2.cvtColor(blured_opt, cv2.COLOR_BGR2HSV_FULL)
        h, s, v = cv2.split(hsv_opt)
        blured_opt = cv2.filter2D(h, -1, kernel)
        blured_opt = cv2.medianBlur(blured_opt, 5, 2)


        # compute saliency
        (success_sar, saliencyMap_sar) = saliency.computeSaliency(blured_sar)
        saliencyMap_sar = (saliencyMap_sar * 255).astype("uint8")

        (success_opt, saliencyMap_opt) = saliency.computeSaliency(blured_opt)
        saliencyMap_opt = (saliencyMap_opt * 255).astype("uint8")

        # SAVE then into proper folders
        cv2.imwrite(os.path.join(sar_saving_path,sar_images[i]), saliencyMap_sar)
        cv2.imwrite(os.path.join(optic_saving_path,fully_name), saliencyMap_opt)

        """
        # Uncomment if you want to compare SSIM metric
        sar_sal.append(saliencyMap_sar)
        opt_sal.append(saliencyMap_opt)
        

        for i in range(len(sar_sal)):
            score = compare_ssim(sar_sal[i], opt_sal[i])
            scores.append(score)
    
        print(scores)
        print("Mean/Std/Var: ", np.mean(scores), np.std(scores), np.var(scores))
        
        plt.plot(scores)
        plt.show()
        """
