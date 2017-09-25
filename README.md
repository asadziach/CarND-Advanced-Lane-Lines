**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./camera_cal/calibration1.jpg "Chessboard"
[image1]: ./camera_cal/test_undist.jpg "Undistorted"
[image2]: ./output_images/undistorted4.jpg "Road Transformed"
[image3]: ./output_images/preprocessed6.jpg "Binary Example"
[image4]: ./output_images/warped5.jpg "Warp Example"
[image5]: ./output_images/drawn5.jpg "Fit Visual"
[image6]: ./output_images/final6.jpg "Output"
[image7]: ./output_images/tracked5.jpg "tracked"
[video1]: ./project_video.mp4 "Video"

---
### Camera Calibration

The code for this step is contained in the file "./CameraCalibration.py" .  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
##### Orginal Image
![alt text][image0]
##### Undistorted Image
![alt text][image1]

### Pipeline (single images)

#### 1. Distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at function preprocesslines #95 through #102 in `./ImageGenerator.py`).  Here's an example of my output for this step. 

![alt text][image3]

#### 3. Perspective transform of image.

The code for my perspective transform includes a function called `birds_eye_perspective()`, which appears in lines 105 through 127 in the file `./ImageGenerator.py`.  The `birds_eye_perspective()` function takes as inputs an image (`image`), calculates source (`src`) and destination (`dst`) points. I chose an ROI used the following constants derived empirically.

```python
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
```

This resulted in the following source and destination points:

| Source             | Destination   |  
|:------------------:|:-------------:|   
| 588.7999   446.399 | 320    0      |
| 691.2000   446.399 | 960    0      |
| 1126.400   673.200 | 960  720      |
| 153.6000   673.200 | 320  720      |


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Identification of lane-line pixels and fit their positions with a polynomial

I used convolution approach with the sliding window method. It maximizes the number of "hot" pixels in each window. A convolution is the summation of the product of two separate signals, in my case the window template and the vertical slice of the pixel image.

I slide my window template across the image from left to right and any overlapping values are summed together, creating the convolved signal. The peak of the convolved signal is where there was the highest overlap of pixels and the most likely position for the lane marker. File `ImageGenerator.py` function get_left_right_centroids(). Here is a debug output of this operation:
![alt text][image7]

The code to fit my lane lines with a 2nd order polynomial is in file `ImageGenerator.py` function fit_lane_lines() line 214:

![alt text][image5]

#### 5. Calculating radius of curvature of the lane and the position of the vehicle with respect to center.

The code is in file `./ImageGenerator.py` functions radius_of_curvature(), fit_lane_lines() and annotate_results() 

#### 6. Results

I implemented this step in my code file `./ImageGenerator.py`  in the function `draw_lane_lines()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

Here's a [link to my video result](./project_video_output.mp4)

#### Tracking and Look-Ahead Filter
Video is a series of related images. I used this to my advantage by keeping track of things like where my last several detections of the lane lines were and what the curvature was. This helps identify outliers in lane detection an provides smoother and more stable results as compared to individual images. This also helps reduce computation costs since I don't need to hunt of lane lines in whole images, instead I simply search within a window around the previous detection.

When I fit a polynomial, then for each y position, I have an x position that represents the lane center from the last frame. I search for the new line within +/- defined margin around the old line center. This is controlled by "hunt" boolean parameter of function get_left_right_centroids() in file 'ImageGenerator.py'


#### Sanity Check
Once my algorithm has found some lines. I check that if the detection makes sense? To confirm that my detected lane lines are real, I implemented the following in function sanity_ok() file VideoGenerator.py:

* Checking that they both lanes have similar curvature
* Checking that they are separated by approximately the right distance horizontally
* Checking that they are roughly parallel

#### Reset
When my sanity checks reveal that the lane lines I've detected are problematic for any reason listed above, I simply assume it was a bad or difficult frame of video. So I retain the previous positions from the frame prior and step to the next frame to search again. If I lose the lines for several frames in a row, I start searching from scratch to re-establish my measurement.

#### Smoothing
Even when everything is working, the line detections tend to jump around from frame to frame a bit. To fix it I smooth it over the last 40 (configurable) frames of video to obtain a cleaner result. Each time I get a new high-confidence measurement, I append it to the list of recent measurements and then take an average over 40(empirical value) past measurements to obtain the lane position that I use to draw onto the image.

---

### Discussion

I tried both the approaches touched in lectures, histogram and convolution. The results are very similar. The most fragile part of the pipeline is the initial creation of binary image by thresholding. I found it challenging to narrow it down to a thresholding range that generalizes well across different videos. By using a specialized convolution kernel, I had success finding lanes on "Challenge" videos. Obviously this will not generalize. The pipeline works best on highways but struggles on back roads and urban areas. There are some other challenges like, road repairs and lighting changes. Some of the techniques to deal with it are distance and curvature sanity testing, smoothing and kalman filters.

One way to fix this is to fit a model of the road instead of looking for just the two lane lines. Lanes lines may be occluded, missing or have dramatic change in gradients, in case of sharp turns. First stage in the pipeline should be to determine what kind of road are we on? If it a forest backroad, then it might be useful to keep track of road boundaries in addition to lane the lines. In difficult frames where lanes lines are lost, we can use road edges to get our bearings and calculate lanes indirectly. A machine learning approach can be more robust than traditional Computer Vision techniques, provided we train it properly.

A deep leanring appraoch can also work. Nvidia trained a [model](https://devblogs.nvidia.com/parallelforall/explaining-deep-learning-self-driving-car/) that is able to drive itself by not only learning the obvious features such as lane markings, but also more subtle features that would be hard to anticipate and program by engineers, for example, bushes lining the edge of the road and atypical vehicle classes. 

On urban roads, Semantic Segmentation by a NN approach, would be more robust in finding the driveable areas, [example](http://www.ri.cmu.edu/pub_files/2016/7/ankit-laddha-ms.pdf).

I tried two Sementic Segmantation implementations with TensorFlow [KittiSeg](https://github.com/MarvinTeichmann/KittiSeg.git) and [dilation-tensorflow](https://github.com/ndrplz/dilation-tensorflow.git). I found them to be slow and needing 'not so moderate' GPU RAM. It does not seem suitable to do per frame. I am not sure if it is an architectural issue with Sementic Segmantation or just that the frameworks I tried have issues.