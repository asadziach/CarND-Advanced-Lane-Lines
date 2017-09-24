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
[image2]: ./test_images/undistorted4.jpg "Road Transformed"
[image3]: ./test_images/preprocessed6.jpg "Binary Example"
[image4]: ./test_images/warped5.jpg "Warp Example"
[image5]: ./test_images/drawn5.jpg "Fit Visual"
[image6]: ./test_images/final6.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file "./CameraCalibration.py" .  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
##### Orginal Image
![alt text][image0]
##### Undistorted Image
![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at function preprocesslines #95 through #102 in `./ImageGenerator.py`).  Here's an example of my output for this step. 

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `birds_eye_perspective()`, which appears in lines 105 through 127 in the file `./ImageGenerator.py`.  The `birds_eye_perspective()` function takes as inputs an image (`image`), calculates source (`src`) and destination (`dst`) points. I chose an ROI used the following constants derviced empirically.

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

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code to fit my lane lines with a 2nd order polynomial is in file `./ImageGenerator.py` function fit_lane_lines() line 214:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code is in file `./ImageGenerator.py` functions radius_of_curvature(), fit_lane_lines() and annotate_results() 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in my code file `./ImageGenerator.py`  in the function `draw_lane_lines()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

Here's a [link to my video result](./project_video_tracked.mp4)

#### Tracking
#### Sanity Check
Ok, so your algorithm found some lines. Before moving on, you should check that the detection makes sense. To confirm that your detected lane lines are real, you might consider:

Checking that they have similar curvature
Checking that they are separated by approximately the right distance horizontally
Checking that they are roughly parallel
#### Look-Ahead Filter
Once you've found the lane lines in one frame of video, and you are reasonably confident they are actually the lines you are looking for, you don't need to search blindly in the next frame. You can simply search within a window around the previous detection.

For example, if you fit a polynomial, then for each y position, you have an x position that represents the lane center from the last frame. Search for the new line within +/- some margin around the old line center.

Double check the bottom of the page here to remind yourself how this works.

Then check that your new line detections makes sense (i.e. expected curvature, separation, and slope).

#### Reset
If your sanity checks reveal that the lane lines you've detected are problematic for some reason, you can simply assume it was a bad or difficult frame of video, retain the previous positions from the frame prior and step to the next frame to search again. If you lose the lines for several frames in a row, you should probably start searching from scratch using a histogram and sliding window, or another method, to re-establish your measurement.

#### Smoothing
Even when everything is working, your line detections will jump around from frame to frame a bit and it can be preferable to smooth over the last n frames of video to obtain a cleaner result. Each time you get a new high-confidence measurement, you can append it to the list of recent measurements and then take an average over n past measurements to obtain the lane position you want to draw onto the image.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I tried both the appraoches touched in lectures, histogram and convolution. The results are very similar. The most fragile part of the pipline is the initial creation of binary image by thresholding. I found it challenging to narrow it down to a thresholding range that generalizes well across different videos. By using a specielized convolution kernel, I succesfully found lanes on "Challenge" videos. Obviously this will not generalize. The pipeline works best on highways but struggles on backroads and urban areas. There are some other challenges like, road repairs and lighting changes. Some of the techniques to deal with it are distance and curvature sanity testing, smoothing and kalman filters. 
One way to fix this is to fit a model of the road instead of looking for just tho two lane lines. Lanes lines may be occluded, missing or have dramatic change in gradients, in case of sharp turns. First stage in the pipleline should be to determine what kind of road are we on? If it a forest backroad, then it might be usefull to keep track of road boundries in addition to lane the lines. In difficult frames where lanes lines are lost, we can use road edges to get our bearings and calculate lanes indirectly.
A machine leaning approach can be more reboust than traditional Computer Vision techniques, provided we train it properly.
A deep leanring appraoch can also work. Nvidia trained a model https://devblogs.nvidia.com/parallelforall/explaining-deep-learning-self-driving-car/ that is able to drive itself by not only learning the obvious features such as lane markings, but also more subtle features that would be hard to anticipate and program by engineers, for example, bushes lining the edge of the road and atypical vehicle classes. 
On urban roads, Sementic Segmantation by a NN pproach, would be more robust in finding the driveable areas http://www.ri.cmu.edu/pub_files/2016/7/ankit-laddha-ms.pdf.
I tried two Sementic Segmantation implementations with TensorFlow https://github.com/MarvinTeichmann/KittiSeg.git and https://github.com/ndrplz/dilation-tensorflow.git. I found them to be slow and needing 'not so moderate' GPU RAM. It does not seem suitable to do per frame. I am not sure if it is an archtecrual issue with Sementic Segmantation or just that the frameworks I tried have issues.