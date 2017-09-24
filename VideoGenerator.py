'''
Created on Sep 23, 2017

@author: asad
'''
from moviepy.editor import VideoFileClip
from ImageGenerator import *
from pywt.data._readers import camera

class VideoLaneProcessor(object):
    
    '''
    class attributes
    '''
    window_size = (25,80)     # Obtained empirically
    dpm = (3.7/700, 30/720) # meters per pixel
    
    def __init__(self, camera_cal_pickle):
        '''
        Constructor
        '''
        dest_pickle = pickle.load( open(camera_cal_pickle, "rb"))
        self.mtx = dest_pickle["mtx"]
        self.dist = dest_pickle["dist"]
        
        self.recent_centers = []
        self.state = LaneState()
            
    def process_image(self, image):
    
        image = undistort(image, self.mtx, self.dist)
    
        preprocessed = preprocess(image)
        
        warped, m_inv = birds_eye_perspective(preprocessed)
        
        window_centroids = get_left_right_centroids(self.state, warped, VideoLaneProcessor.window_size, 
                                                    margin=25)
        
        lanes, yvals, camera_center = fit_lane_lines(self.state, image.shape[0], window_centroids, 
                                                     VideoLaneProcessor.window_size, smoothing=40)
        
        curve_radii = radius_of_curvature(image.shape[0],VideoLaneProcessor.dpm,window_centroids, yvals)
        
        result, _ = draw_lane_lines(image, m_inv, lanes, colors=([0,255,0],[0,255,0]))
        
        annotate_results(result, camera_center, VideoLaneProcessor.dpm, curve_radii)
    
        return result  
        
def main():
    output = 'project_video_tracked.mp4'
    input  = 'project_video.mp4'
    
    clip = VideoFileClip(input)
    processor = VideoLaneProcessor("camera_cal/wide_dist_pickle.p")
    video_clip = clip.fl_image(processor.process_image)
    video_clip.write_videofile(output, audio=False)
    
if __name__ == '__main__':
    main()