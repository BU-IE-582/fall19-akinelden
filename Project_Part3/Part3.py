from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d

plt.gray()
window_shape = (24, 24)
step_length = 8
bin_number = 16

def getPatches(pixel_array, window_shape, step_length):
    edge_size = pixel_array.shape[0]
    whole_data = extract_patches_2d(pixel_array, window_shape)
    last_patch_index = edge_size - window_shape[0]
    indices = np.array(range(0,last_patch_index,step_length))
    desired_patch_indices = np.array([])
    for i in indices:
        desired_patch_indices = np.concatenate((desired_patch_indices, indices + i*last_patch_index))
    desired_patch_indices = np.array(desired_patch_indices, dtype="int32")
    patches = whole_data[desired_patch_indices]
    return patches

def getPixelRatios(pixel_array, bin_number):
    bin_size = int(256 / bin_number)
    shape = pixel_array.shape
    flattened = pixel_array.reshape((shape[0],-1))
    range_data = np.array(flattened / bin_size, dtype="int32")
    pixel_freqs = np.apply_along_axis(lambda arr: np.bincount(arr, minlength=bin_number), arr=range_data, axis=1)
    pixel_ratios = pixel_freqs / flattened.shape[1]
    return pixel_ratios

def calculateBinLimits(patch_ratios, sigmalevel=3):
    xbarbar = patch_ratios.mean(axis=0)
    sigma = patch_ratios.std(axis=0)
    sigma[xbarbar <= 0.001] *= 2
    upper_limit = xbarbar + sigmalevel * sigma
    lower_limit = xbarbar - sigmalevel * sigma
    lower_limit[lower_limit < 0] = 0
    #upper_limit[xbarbar <= 0.001] *= 2
    return (xbarbar, upper_limit, lower_limit, sigma)

def getOutlierPatchIndices(pixel_ratios, bin_limits, window_shape, step_length, bin_number):
    patch_number = int((512-window_shape[0])/step_length)**2
    outlier_score=np.zeros((patch_number,bin_number))

    for bin_no in range(bin_number):
        bin_density = pixel_ratios[:,bin_no]
        outlier_indices_up=np.where(bin_density > bin_limits[1][bin_no])[0]
        outlier_score[outlier_indices_up, bin_no] += (pixel_ratios[outlier_indices_up, bin_no] - bin_limits[1][bin_no]) / bin_limits[3][bin_no]
        outlier_indices_down = np.where( bin_density < bin_limits[2][bin_no] )[0]
        outlier_score[outlier_indices_down, bin_no] += (bin_limits[2][bin_no] - pixel_ratios[outlier_indices_down, bin_no]) / bin_limits[3][bin_no]
    sum_score = np.sum(outlier_score, axis=1)
    indices = np.where(sum_score > 2)[0]
    return (indices, sum_score[indices])

def plotXBarChart(patch_ratios, bin_limits, figsize=[16,20]):
    xbarbar = bin_limits[0]
    upper_limit = bin_limits[1]
    lower_limit = bin_limits[2]

    bin_number = patch_ratios.shape[1]
    plt.figure(figsize=figsize)
    for i in range(bin_number):
        plt.subplot(bin_number/2,2,i+1)
        plt.title("Pixel Ratio X-Bar Chart - Bin #{0}".format(i))
        plt.xlabel('Patch number')
        plt.ylabel('Pixel Ratio (X-Bar)')
        plt.plot(patch_ratios[:,i])
        plt.axhline(upper_limit[i],color="r", linestyle='--', label='UCL/LCL')
        plt.axhline(lower_limit[i],color="r", linestyle='--')
        plt.axhline(xbarbar[i],color="g", label='CL')
        plt.legend()
        plt.tight_layout()
    plt.show()
    
def fillOutliers(image, outlier_patch_infos, window_shape, step_length, figsize=[10,16], alpha=0.15):
    total_patch_in_row = int( (image.shape[0] - window_shape[0]) / step_length ) + 1
    outlier_patch_indices = outlier_patch_infos[0]
    plt.figure(figsize=figsize)
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.tight_layout()
    plt.subplot(1,2,2)
    plt.imshow(image)
    origin_x = ( outlier_patch_indices % total_patch_in_row ) * step_length
    origin_y = np.array((outlier_patch_indices / total_patch_in_row) * step_length, dtype='uint32')
    for i in range(len(outlier_patch_indices)):
        corner_xs = [origin_x[i],
                  origin_x[i]+window_shape[0],
                  origin_x[i]+window_shape[0],
                  origin_x[i]]
        corner_ys = [origin_y[i],
                  origin_y[i],
                  origin_y[i]+window_shape[0],
                  origin_y[i]+window_shape[0]]
        plt.fill(corner_xs, corner_ys, 'r', alpha=min(1,alpha * outlier_patch_infos[1][i]))
    plt.tight_layout()
    plt.show()