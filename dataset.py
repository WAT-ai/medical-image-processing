import os
import glob
import json
import numpy as np

IMAGE_TYPES = ('ADC', 'DWI', 'Ktrans', 'T2w', 'CDIs')


class LesionDataset:
    """Main dataset class for PROSTATEx/PROSTATEx-2"""
    def __init__(self, data_dir):
        """
        Parameters
        ----------
        data_dir : str
            Path to the prepared data directory.
        """
        self.data_dir = data_dir
        self._load_lesion_info()

    def query(self, patient_id, cache=True):
        patient_dict = self._data_dict[patient_id]
        return Patient(patient_id, patient_dict, cache=cache)

    def patients(self, cache=False):
        for patient_id, patient_dict in self._data_dict.items():
            yield Patient(patient_id, patient_dict, cache=cache)

    def _load_lesion_info(self):
        self._data_dict = {}
        for patient_dir in glob.glob(os.path.join(self.data_dir, 'ProstateX-*')):
            # Get ID
            patient_id = os.path.basename(patient_dir)

            # Get image and mask files
            img_dir = os.path.join(patient_dir, 'images')
            lesion_mask_dir = os.path.join(patient_dir, 'lesion_masks')
            prostate_mask_dir = os.path.join(patient_dir, 'prostate_masks')
            image_files = {
                name: os.path.join(img_dir, name + '.npy') for name in IMAGE_TYPES}
            lesion_mask_files = {
                name: os.path.join(lesion_mask_dir, name + '.npy')
                for name in IMAGE_TYPES[:1] + IMAGE_TYPES[2:4]}
            lesion_mask_files['DWI'] = lesion_mask_files['ADC']
            prostate_mask_files = {
                name: os.path.join(prostate_mask_dir, name + '.npy')
                for name in IMAGE_TYPES[:1] + IMAGE_TYPES[2:4]}
            prostate_mask_files['DWI'] = prostate_mask_files['ADC']

            # Load additional image data
            params_file = os.path.join(patient_dir, 'params.json')
            with open(params_file, 'r') as f:
                params = json.load(f)
            spacings = {key: np.array(val) for key, val in params['spacings'].items()}
            world_matrices = {key: np.array(val) for key, val in params['world_matrices'].items()}
            bvals = np.array(params['b_vals'])

            self._data_dict[patient_id] = {
                'images': image_files,
                'lesion_masks': lesion_mask_files,
                'prostate_masks': prostate_mask_files,
                'spacings': spacings,
                'world_matrices': world_matrices,
                'b_vals': bvals
            }

    def __len__(self):
        return len(self._data_dict)


class Patient:
    def __init__(self, patient_id, patient_dict, cache=False):
        self.patient_id = patient_id
        self._patient_dict = patient_dict
        self._cache = cache
        self._init_cache()

    def slice_data(self, image_type, slice_idx, crop_to_mask=False, pad=0):
        image, prostate_mask, lesion_mask = self.image_data(image_type)
        image = image[..., slice_idx]
        prostate_mask = prostate_mask[..., slice_idx]
        lesion_mask = lesion_mask[..., slice_idx]
        if crop_to_mask:
            return self.crop_to_mask(image, prostate_mask, lesion_mask, self.spacing(image_type), pad=pad)
        return image, prostate_mask, lesion_mask

    def image_data(self, image_type):
        if image_type in ('DWI', 'CDIs'):
            return self._load_images(image_type, mask_type='ADC')
        return self._load_images(image_type)

    def world_matrix(self, image_type):
        if image_type in ('DWI', 'CDIs'):
            image_type = 'ADC'
        return self._patient_dict['world_matrices'][image_type]

    def spacing(self, image_type):
        if image_type in ('DWI', 'CDIs'):
            image_type = 'ADC'
        return self._patient_dict['spacings'][image_type]

    def clear_cache(self):
        self._init_cache()

    def bvals(self):
        return self._patient_dict['b_vals']

    @staticmethod
    def crop_to_mask(image, prostate_mask, lesion_mask, spacing, pad=0):
        y, x = np.where((prostate_mask > 0) | (lesion_mask > 0))
        padx = int(round(pad/spacing[0]))
        pady = int(round(pad/spacing[1]))
        xmin = max(x.min() - padx, 0)
        ymin = max(y.min() - pady, 0)
        xmax = min(x.max() + 1 + padx, image.shape[1])
        ymax = min(y.max() + 1 + pady, image.shape[0])

        image = image[ymin:ymax, xmin:xmax]
        lesion_mask = lesion_mask[ymin:ymax, xmin:xmax]
        prostate_mask = prostate_mask[ymin:ymax, xmin:xmax]

        return image, prostate_mask, lesion_mask

    def _init_cache(self):
        self._images = {
            image_type: None for image_type in IMAGE_TYPES}
        self._lesion_masks = {
            image_type: None for image_type in IMAGE_TYPES[:1] + IMAGE_TYPES[2:4]}
        self._prostate_masks = {
            image_type: None for image_type in IMAGE_TYPES[:1] + IMAGE_TYPES[2:4]}

    def _load_images(self, image_type, mask_type=None):
        mask_type = mask_type if mask_type is not None else image_type
        if self._images[image_type] is None:
            # Load image
            image_file = self._patient_dict['images'][image_type]
            image = np.load(image_file)

            # Move DWI b-value dimension
            if image_type == 'DWI':
                image = np.moveaxis(image, 0, 2)

            # Load masks
            if self._lesion_masks[mask_type] is None:
                lesion_mask_file = self._patient_dict['lesion_masks'][mask_type]
                prostate_mask_file = self._patient_dict['prostate_masks'][mask_type]

                lesion_mask = np.load(lesion_mask_file)
                prostate_mask = np.load(prostate_mask_file)
            else:
                lesion_mask = self._lesion_masks[mask_type]
                prostate_mask = self._prostate_masks[mask_type]

            # Store if caching is enabled
            if self._cache:
                self._images[image_type] = image
                self._lesion_masks[mask_type] = lesion_mask
                self._prostate_masks[mask_type] = prostate_mask
        else:
            # Use stored values
            image = self._images[image_type]
            lesion_mask = self._lesion_masks[mask_type]
            prostate_mask = self._prostate_masks[mask_type]

        return image, prostate_mask, lesion_mask


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from data.PreparedPROSTATExDataset.utils import plot_mask

    data_dir = 'F:\\Datasets\\PROSTATEx\\final_dataset_full_vol_final\\data'
    query_patient = 'ProstateX-0011'
    slice_index = 11

    dataset = LesionDataset(data_dir)
    patient = dataset.query(query_patient)

    adc, adc_prost_mask, adc_mask = patient.slice_data('ADC', slice_index)
    t2w, t2w_prost_mask, t2w_mask = patient.slice_data('T2w', slice_index)
    ktrans, ktrans_prost_mask, ktrans_mask = patient.slice_data('Ktrans', slice_index)
    cdis, cdis_prost_mask, cdis_mask = patient.slice_data('CDIs', slice_index)

    _, ax = plt.subplots(2, 2)
    ax = ax.ravel()
    ax[0].imshow(adc, cmap='gray')
    plot_mask(ax[0], adc_prost_mask, color='g', linestyle='-')
    plot_mask(ax[0], adc_mask, color='r', linestyle='-')
    ax[0].set_title('ADC')
    ax[1].imshow(t2w, cmap='gray')
    plot_mask(ax[1], t2w_prost_mask, color='g', linestyle='-')
    plot_mask(ax[1], t2w_mask, color='r', linestyle='-')
    ax[1].set_title('T2w')
    ax[2].imshow(ktrans, cmap='gray')
    plot_mask(ax[2], ktrans_prost_mask, color='g', linestyle='-')
    plot_mask(ax[2], ktrans_mask, color='r', linestyle='-')
    ax[2].set_title('$k^{trans}$')
    ax[3].imshow(cdis, cmap='gray')
    plot_mask(ax[3], cdis_prost_mask, color='g', linestyle='-')
    plot_mask(ax[3], cdis_mask, color='r', linestyle='-')
    ax[3].set_title('CDIs')

    adc, adc_prost_mask, adc_mask = patient.slice_data('ADC', slice_index, crop_to_mask=True)
    t2w, t2w_prost_mask, t2w_mask = patient.slice_data('T2w', slice_index, crop_to_mask=True)
    ktrans, ktrans_prost_mask, ktrans_mask = patient.slice_data('Ktrans', slice_index, crop_to_mask=True)
    cdis, cdis_prost_mask, cdis_mask = patient.slice_data('CDIs', slice_index, crop_to_mask=True)

    _, ax = plt.subplots(2, 2)
    ax = ax.ravel()
    ax[0].imshow(adc, cmap='gray')
    plot_mask(ax[0], adc_prost_mask, color='g', linestyle='-')
    plot_mask(ax[0], adc_mask, color='r', linestyle='-')
    ax[0].set_title('ADC')
    ax[1].imshow(t2w, cmap='gray')
    plot_mask(ax[1], t2w_prost_mask, color='g', linestyle='-')
    plot_mask(ax[1], t2w_mask, color='r', linestyle='-')
    ax[1].set_title('T2w')
    ax[2].imshow(ktrans, cmap='gray')
    plot_mask(ax[2], ktrans_prost_mask, color='g', linestyle='-')
    plot_mask(ax[2], ktrans_mask, color='r', linestyle='-')
    ax[2].set_title('$k^{trans}$')
    ax[3].imshow(cdis, cmap='gray')
    plot_mask(ax[3], cdis_prost_mask, color='g', linestyle='-')
    plot_mask(ax[3], cdis_mask, color='r', linestyle='-')
    ax[3].set_title('CDIs')
    plt.show()
