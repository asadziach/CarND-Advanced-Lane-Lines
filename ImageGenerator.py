'''
Created on Sep 18, 2017

@author: asad
'''
import numpy as np
import cv2
import glob
import pickle
import glob
from tracker import tracker

# Define a function that thresholds the S-channel of HLS
def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

# Define a function that thresholds the S-channel of HLS
def hsv_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hls[:,:,2]
    binary_output = np.zeros_like(v_channel)
    binary_output[(v_channel > thresh[0]) & (v_channel <= thresh[1])] = 1
    return binary_output

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),
           max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def main():
    dest_pickle = pickle.load( open("camera_cal/wide_dist_pickle.p", "rb"  ))
    mtx = dest_pickle["mtx"]
    dist = dest_pickle["dist"]

    images = glob.glob( './test_images/test*.jpg' )
    
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        img = cv2.undistort(img, mtx, dist, None, mtx)
        
        preprocessImage = np.zeros_like(img[:,:,0])
        gradx = abs_sobel_thresh(img, orient='x', thresh=(12,255))
        grady = abs_sobel_thresh(img, orient='x', thresh=(25,255))
        hls   = hls_select(img, thresh=(100,255))
        hsv   = hsv_select(img, thresh=(50,255))
        preprocessImage[(gradx==1) & (grady==1) | (hls==1) & (hsv==1) ] = 255
        
        img_size = (img.shape[1], img.shape[0])
        # Obtained empirically to map trapezoid to birds eye view
        bot_width = 0.76
        mid_width = 0.08
        height_pct = 0.62
        bottom_trim = 0.935
        
        src = np.float32([[img.shape[1]*(.5-mid_width/2),img.shape[0]*height_pct],
                          [img.shape[1]*(.5+mid_width/2),img.shape[0]*height_pct],
                          [img.shape[1]*(.5+bot_width/2),img.shape[0]*bottom_trim],
                          [img.shape[1]*(.5-bot_width/2),img.shape[0]*bottom_trim]])
        offset = img_size[0]*.25
        dst = np.float32([[offset, 0], [img_size[0]-offset, 0], 
                          [img_size[0]-offset, img_size[1]],
                          [offset, img_size[1]]])
        
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        wrapped = cv2.warpPerspective(preprocessImage, M, img_size, flags=cv2.INTER_LINEAR)
        
        # Obtained empirically
        window_width =25
        window_height = 80
        curve_centers = tracker(window_size=(window_width,window_height),window_margin=25,
                                dpm=(10/720, 4/384),smoothing=15)
        
        window_centroids = curve_centers.get_left_right_centroids(wrapped)
        
        l_points = np.zeros_like(wrapped)
        r_points = np.zeros_like(wrapped)
        
        for level in range(0,len(window_centroids)):
            l_mask = window_mask(window_width,window_height,wrapped,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,wrapped,window_centroids[level][1],level)
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255
        
        # temp visual debug
        template = np.array(r_points+l_points,np.uint8) 
        zero_channel = np.zeros_like(template)  
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) 
        warpage = np.array(cv2.merge((wrapped,wrapped,wrapped)),np.uint8) 
        result = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) 
         
        write_file = './test_images/tracked' + str(idx) + '.jpg'
        cv2.imwrite(write_file, result)
            
if __name__ == '__main__':
    main()