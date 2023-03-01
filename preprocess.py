import os
import numpy as np
import matplotlib.pyplot as plt

def flatten_image(path_to_dataset):
    """"""
    for patient in os.listdir(path_to_dataset):
        # gather images for ADC, CDIs, and DWI modalities
        adc = np.load(os.path.join(path_to_dataset, f"ProstateX-{patient_num:04}\images\ADC.npy"))[:, :, :]
        cdis = np.load(os.path.join(path_to_dataset, f"ProstateX-{patient_num:04}\images\CDIs.npy"))[:, :, :]
        dwi = np.load(os.path.join(path_to_dataset, f"ProstateX-{patient_num:04}\images\DWI.npy"))[0, :, :, :]

        # stack images and channels into 2D, reshape into three dimensions
        total_slice = adc.shape[2] + cdis.shape[2] + dwi.shape[2]
        img_stacked = np.stack((adc, cdis, dwi), axis=3)
        img_reshape = np.reshape(img_stacked, (128, 84, total_slice))

    return img_reshape

def data_visualizer(path_to_dataset, patient_num, slice):
    """"""
    img_reshape = flatten_image(path_to_dataset)
    # Assuming that img_reshape is the final reshaped image
    if slice == all:
        min_slice = 0
        while min_slice < total_slice - 1:
            print(f'slice {min_slice}')
            plt.imshow(img_reshape[:,:,min_slice], cmap='gray')
            plt.show()
            min_slice = min_slice + 1
                
    else:
        plt.imshow(img_reshape[:,:,slice], cmap='gray')
        plt.show()