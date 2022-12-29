import cv2
import numpy as np
from matplotlib import cm


def convert_mask(mask, zval, src_mat, dst_mat, out_shape):
    """Converts a mask to the coordinate space of a different image series"""
    contours = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0]
    mask_out = np.zeros(out_shape, dtype=np.uint8)
    hom_z = np.ones(4)
    hom_z[2] = zval
    z = int(round(np.dot(np.linalg.inv(dst_mat), np.dot(src_mat, hom_z))[2]))
    for cnt in contours:
        xy = cnt[:, 0, :].T
        dst_coords = convert_pixel_coords(xy, src_mat, dst_mat, z=zval)
        new_cnt = dst_coords[:2, :].T[:, None, :]
        cv2.drawContours(mask_out, [new_cnt], 0, 1, -1)
    return mask_out, z


def convert_pixel_coords(xy, src_mat, dst_mat, z=1):
    """Converts coordinates from one image space to another"""
    zh = np.ones(xy.shape)
    zh[1, :] *= z
    hom_coords = np.concatenate((xy, zh), axis=0).astype(np.float64)
    world_coords = np.dot(src_mat, hom_coords)
    dst_coords = np.dot(np.linalg.inv(dst_mat), world_coords)
    dst_coords = np.round(dst_coords[:3, :]).astype(np.int32)
    return dst_coords


def find_contours(binary_image):
    """Helper function for finding contours"""
    return cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0]


def plot_mask(ax, mask, **kwargs):
    """Helper function to plot a contour outline"""
    if mask.dtype == np.bool:
        mask = mask.astype(np.uint8)
    contours = find_contours(mask)
    label = kwargs.pop('label', None)
    labelled = False
    for contour in contours:
        contour = contour[:, 0, :]
        contour = np.concatenate((contour, contour[:1]), axis=0)
        if labelled:
            ax.plot(contour[:, 0], contour[:, -1], **kwargs)
        else:
            ax.plot(contour[:, 0], contour[:, -1], label=label, **kwargs)
            labelled = True


def heatmap_overlay(image, heatmap, mask=None, alpha=0.2, cmap='jet'):
    """Makes colour-mapped overlay of a heatmap on an image"""
    if image.ndim < 3:
        image = image[..., None]
    if mask.ndim < 3:
        mask = mask[..., None]
    heatmap = make_heatmap(heatmap, cmap=cmap)
    overlay = alpha_blend(heatmap, image, alpha, mask=mask)
    return overlay


def make_heatmap(mask, cmap='jet'):
    """Converts a mask into a heatmap"""
    # Get colormap indices
    indices = np.round(255.*mask).astype(np.int32)

    # Get colourmap values
    levels = list(range(256))
    cm_func = cm.get_cmap(cmap)
    cmap_vals = cm_func(levels)[:, :3]

    # Gather colourmap values at indices
    heatmap = np.take(cmap_vals, indices, axis=0)

    return heatmap


def alpha_blend(image1, image2, alpha=0.2, mask=None):
    """Alpha blends two images"""
    if mask is None:
        alpha_mask = alpha
    else:
        alpha_mask = alpha*mask.astype(np.float32)
    image1 = image1 if isinstance(image1, np.floating) else image1.astype(np.float32)
    image2 = image2 if isinstance(image2, np.floating) else image2.astype(np.float32)
    return alpha_mask*image1 + (1. - alpha_mask)*image2
