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
    window_size = (50,100)     # Obtained empirically
    dpm = (3.7/700, 30/720)   # meters per pixel
    
    min_lane_distance = 425   # pixles 
    max_lane_distance = 550    # pixels, challenge vidoe does better with 500
    
    curve_tolerance  = 600
    bad_frame_threshold = 10
    lane_smoothing = 3
    drift_cotrol = 10
    margin = 25
    curve_radii_smoothing = 3
    
    def __init__(self, camera_cal_pickle):
        '''
        Constructor
        '''
        dest_pickle = pickle.load( open(camera_cal_pickle, "rb"))
        self.mtx = dest_pickle["mtx"]
        self.dist = dest_pickle["dist"]
        
        self.recent_lanes = []
        self.recent_centers = []
        self.recent_curve_radii = []
        self.needs_reset = True
        self.bad_frame_count = 0
        self.frame_count = 0   
     
    '''
        * Checking that they both lanes have similar curvature
        * Checking if they are separated by the right distance horizontally
        * Checking that they are roughly parallel     
    '''           
    def sanity_ok(self, window_centroids, curve_radii):
        left_center = window_centroids[:,0][0]
        right_center = window_centroids[:,1][0]
    
        lane_distance = (right_center - left_center)
        if (lane_distance < VideoLaneProcessor.min_lane_distance or 
            lane_distance > VideoLaneProcessor.max_lane_distance):
            return  False
        
        left_roc, right_roc = curve_radii
        diffence = abs(left_roc - right_roc)
        if diffence > VideoLaneProcessor.curve_tolerance:
            return  False
        return True
    
    # Video processing pipeline
    def process_image(self, image):
    
        image = undistort(image, self.mtx, self.dist)
    
        preprocessed = create_binary_image(image)
        
        warped, m_inv = birds_eye_perspective(preprocessed)
        
        window_centroids = get_left_right_centroids(self.recent_centers, warped, 
                                                    VideoLaneProcessor.window_size, 
                                                    margin=VideoLaneProcessor.margin, 
                                                    hunt=self.needs_reset)
        
        lanes, yvals, camera_center = fit_lane_lines(image.shape[0], window_centroids, 
                                                     VideoLaneProcessor.window_size)
        
        curve_radii = radius_of_curvature(image.shape[0],VideoLaneProcessor.dpm,
                                                    window_centroids, yvals)
        
        if not self.sanity_ok(window_centroids, curve_radii):
            #bad frame
            if self.frame_count == 0:
                # Don't do anything if frist frame is bad
                return image
            lanes = self.recent_lanes[-1]
            curve_radii = self.recent_curve_radii[-1]
            self.bad_frame_count += 1         
        else: # Good frame
            self.recent_lanes.append(lanes)
            self.recent_curve_radii.append(curve_radii)
            self.needs_reset = False

            
        lanes = np.average(self.recent_lanes[-VideoLaneProcessor.lane_smoothing:], axis=0)
        curve_radii = np.average(self.recent_curve_radii[-VideoLaneProcessor.curve_radii_smoothing:], 
                                 axis=0)

        result, _ = draw_lane_lines(image, m_inv, lanes, colors=([0,255,0],[0,255,0]))
        
        annotate_results(result, camera_center, VideoLaneProcessor.dpm, curve_radii)
    
        self.frame_count += 1
        
        # Conditions for Reset
        if (self.frame_count < VideoLaneProcessor.lane_smoothing or 
            self.frame_count < VideoLaneProcessor.drift_cotrol        or
            self.bad_frame_count > VideoLaneProcessor.bad_frame_threshold):
            self.needs_reset = True 
        
        return result  
        
def main():
    videoname = 'project_video'
    output = videoname + '_output.mp4'
    input  = videoname + '.mp4'
    
    clip = VideoFileClip(input)
    processor = VideoLaneProcessor("camera_cal/wide_dist_pickle.p")
    video_clip = clip.fl_image(processor.process_image)
    video_clip.write_videofile(output, audio=False)
    
if __name__ == '__main__':
    main()
