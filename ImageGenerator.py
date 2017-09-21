'''
Created on Sep 18, 2017

@author: asad
'''
import numpy as np
import cv2
import glob
import pickle

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

def undistort(image, mtx, dist):
    img = cv2.imread(image)
    img = cv2.undistort(img, mtx, dist, None, mtx)
    return img
    
def preprocess(image):
    preprocessImage = np.zeros_like(image[:,:,0])
    gradx = abs_sobel_thresh(image, orient='x', thresh=(12,255))
    grady = abs_sobel_thresh(image, orient='y', thresh=(25,255))
    hls   = hls_select(image, thresh=(100,255))
    hsv   = hsv_select(image, thresh=(50,255))
    preprocessImage[(gradx==1) & (grady==1) | (hls==1) & (hsv==1) ] = 255    
    return preprocessImage

def birds_eye_perspective(image):
    # Obtained empirically to map trapezoid to birds eye view
    bot_width = 0.76
    mid_width = 0.08
    height_pct = 0.62
    bottom_trim = 0.935
    
    img_size = (image.shape[1], image.shape[0])
            
    src_img = np.float32([[img_size[0]*(.5-mid_width/2),img_size[1]*height_pct],
                      [img_size[0]*(.5+mid_width/2),img_size[1]*height_pct],
                      [img_size[0]*(.5+bot_width/2),img_size[1]*bottom_trim],
                      [img_size[0]*(.5-bot_width/2),img_size[1]*bottom_trim]])
    offset = img_size[0]*.25
    dst_img = np.float32([[offset, 0], [img_size[0]-offset, 0], 
                      [img_size[0]-offset, img_size[1]],
                      [offset, img_size[1]]])
    
    M = cv2.getPerspectiveTransform(src_img, dst_img)
    Minv = cv2.getPerspectiveTransform(dst_img, src_img)
    wrapped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
    
    return wrapped, Minv      

def get_left_right_centroids(image, window_size):
    
    # Obtained empirically
    margin = 25
    smoothing = 15
    
    width = window_size[0]
    height = window_size[1]
    recent_centers = []

    centroids = []
    window = np.ones(width)
    
    #Left
    left_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    left_center = np.argmax(np.convolve(window,left_sum)) - width/2
    #Right
    right_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    right_center = np.argmax(np.convolve(window,right_sum)) - width/2 + int(image.shape[1]/2)
    
    centroids.append((left_center, right_center))
    
    # Move up
    for level in range(1,(int)(image.shape[0]/height)):
       
        image_layer = np.sum(image[int(image.shape[0]-(level+1)*height):
                                    int(image.shape[0]-level*height),:], axis=0)
        
        conv_signal = np.convolve(window, image_layer)
        
        offset = width/2
        
        left_min_index = int(max(left_center+offset-margin, 0))
        left_max_index = int(min(left_center+offset+margin, image.shape[1]))
        left_center = np.argmax(conv_signal[left_min_index:left_max_index])+left_min_index-offset
        
        right_min_index = int(max(right_center+offset-margin, 0))
        right_max_index = int(min(right_center+offset+margin, image.shape[1]))
        right_center = np.argmax(conv_signal[right_min_index:right_max_index])+right_min_index-offset
        
        centroids.append((left_center,right_center))
        
    recent_centers.append(centroids)
    
    # return result averaged over past centers.
    return np.average(recent_centers[-smoothing:], axis=0)

def draw_visual_debug(image, window_centroids, window_size):
    l_points = np.zeros_like(image)
    r_points = np.zeros_like(image)
    
    window_width = window_size[0]
    window_height = window_size[1]
            
    for level in range(0,len(window_centroids)):
        l_mask = window_mask(window_width,window_height,image,window_centroids[level][0],level)
        r_mask = window_mask(window_width,window_height,image,window_centroids[level][1],level)
        l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
        r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255
    
    # temp visual debug
    template = np.array(r_points+l_points,np.uint8) 
    zero_channel = np.zeros_like(template)  
    template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) 
    warpage = np.array(cv2.merge((image,image,image)),np.uint8) 
    return cv2.addWeighted(warpage, 1, template, 0.5, 0.0) 
            
def main():
    dest_pickle = pickle.load( open("camera_cal/wide_dist_pickle.p", "rb"  ))
    mtx = dest_pickle["mtx"]
    dist = dest_pickle["dist"]

    images = glob.glob( './test_images/test*.jpg' )
    
    for idx, fname in enumerate(images):

        image = undistort(fname, mtx, dist)
        #Debug point
        cv2.imwrite('./test_images/undistorted' + str(idx) + '.jpg', image)
        
        preprocessed = preprocess(image)
        
        #Debug point
        cv2.imwrite('./test_images/preprocessed' + str(idx) + '.jpg', preprocessed)
                
        wrapped, _ = birds_eye_perspective(preprocessed)

        #Debug point
        cv2.imwrite('./test_images/wrapped' + str(idx) + '.jpg', wrapped)
        
        window_size = (25,80)     # Obtained empirically
        window_centroids = get_left_right_centroids(wrapped, window_size)   
        tracked = draw_visual_debug(wrapped, window_centroids, window_size)
         
        cv2.imwrite('./test_images/tracked' + str(idx) + '.jpg', tracked)
        
if __name__ == '__main__':
    main()