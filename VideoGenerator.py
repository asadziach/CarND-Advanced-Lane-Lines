'''
Created on Sep 23, 2017

@author: asad
'''
from moviepy.editor import VideoFileClip
from ImageGenerator import *

dest_pickle = pickle.load( open("camera_cal/wide_dist_pickle.p", "rb"  ))
mtx = dest_pickle["mtx"]
dist = dest_pickle["dist"]

def process_image(image):

    image = undistort(image, mtx, dist)

    preprocessed = preprocess(image)
    
    wrapped, m_inv = birds_eye_perspective(preprocessed)
    
    window_size = (25,80)     # Obtained empirically
    window_centroids = get_left_right_centroids(wrapped, window_size)
    
    lanes, yvals, camera_center = fit_lane_lines(image.shape[0], window_centroids, window_size)
    result, _ = draw_lane_lines(image, m_inv, lanes, colors=([255,0,0],[0,0,255]))
    
    dpm = (10/720, 4/384)
    curve_radii = radius_of_curvature(image.shape[0],dpm,window_centroids, yvals)
    
    annotate_results(result, camera_center, dpm, curve_radii)

    return result  
        
def main():
    output = 'project_videoo_tracked.mp4'
    input  = 'project_video.mp4'
    
    clip = VideoFileClip(input)
    video_clip = clip.fl_image(process_image)
    video_clip.write_videofile(output, audio=False)
    
if __name__ == '__main__':
    main()