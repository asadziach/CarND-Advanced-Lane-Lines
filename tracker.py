'''
Created on Sep 17, 2017

@author: asad
'''
import numpy as np

class tracker():
    '''
    dpm = pixels per meter
    smoothing = average over past outputs
    '''
    def __init__(self, window_size, window_margin, dpm, smoothing=15):
        
        self.recent_centers = []
        
        self.window_size = window_size
        
        self.margin = window_margin
        
        self.dpm = dpm
        
        self.smoothing = smoothing
        
    '''
    img = distortion corrected birds eye view image of road.
    '''
    def get_left_right_centroids(self, img):
        
        margin = self.margin
        window_width = self.window_size[0]
        window_height = self.window_size[1]
        
        window_centroids = []
        window = np.ones(window_width)
        
        #Left
        l_sum = np.sum(img[int(3*img.shape[0]/4):,:int(img.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum)) - window_width/2
        #Right
        r_sum = np.sum(img[int(3*img.shape[0]/4):,int(img.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum)) - window_width/2 + int(img.shape[1]/2)
        
        window_centroids.append((l_center, r_center))
        
        # Move up
        for level in range(1,(int)(img.shape[0]/window_height)):
           
            image_layer = np.sum(img[int(img.shape[0]-(level+1)*window_height):
                                        int(img.shape[0]-level*window_height),:], axis=0)
            
            conv_signal = np.convolve(window, image_layer)
            
            offset = window_width/2
            
            l_min_index = int(max(l_center+offset-margin, 0))
            l_max_index = int(min(l_center+offset+margin, img.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            
            r_min_index = int(max(r_center+offset-margin, 0))
            r_max_index = int(min(r_center+offset+margin, img.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
            
            window_centroids.append((l_center,r_center))
            
        self.recent_centers.append(window_centroids)
        
        # return result averaged over past centers.
        return np.average(self.recent_centers[-self.smoothing:], axis=0)
        