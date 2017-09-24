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
    dpm = (3.7/700, 30/720)   # meters per pixel
    
    min_lane_distance = 507   # pixles (U.S. regulations)
    max_lane_distane = 659   # pixels
    
    def __init__(self, camera_cal_pickle):
        '''
        Constructor
        '''
        dest_pickle = pickle.load( open(camera_cal_pickle, "rb"))
        self.mtx = dest_pickle["mtx"]
        self.dist = dest_pickle["dist"]
        
        self.state = LaneState()
        self.recent_curve_radii = []
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None        
            
    def sanity_ok(self, window_centroids, curve_radii):
        left_center = window_centroids[:,0][0]
        right_center = window_centroids[:,1][0]
    
        lane_distance = (right_center - left_center)
        if lane_distance < VideoLaneProcessor.min_lane_distance or lane_distance > VideoLaneProcessor.max_lane_distane:
            return  False
        return True
    
    def process_image(self, image):
    
        image = undistort(image, self.mtx, self.dist)
    
        preprocessed = preprocess(image)
        
        warped, m_inv = birds_eye_perspective(preprocessed)
        
        window_centroids = get_left_right_centroids(self.state, warped, VideoLaneProcessor.window_size, 
                                                    margin=25)
        
        lanes, yvals, camera_center = fit_lane_lines(self.state, image.shape[0], window_centroids, 
                                                     VideoLaneProcessor.window_size, smoothing=1)
        
        curve_radii = radius_of_curvature(image.shape[0],VideoLaneProcessor.dpm,window_centroids, yvals)
        
        if not self.sanity_ok(window_centroids, curve_radii):
            lanes = self.state.recent_lanes[-1]
        else:
            self.state.recent_lanes.append(lanes)
            lanes = np.average(self.state.recent_lanes[-40:], axis=0)

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