# Importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# Display plots inline in iPython
#%matplotlib inline

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    #return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Previous values characterizing the last drawn left line
prevMLeftLine = None
prevBLeftLine = None
prevX1LeftLine = None
prevX2Leftline = None

# Previous values characterizing the last drawn right line
prevMRightLine = None
prevBRightLine = None
prevX1RightLine = None
prevX2Rightline = None

def draw_lines(img, lines, color=[255, 0, 0], thickness=8):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    
    global prevMLeftLine
    global prevBLeftLine
    global prevX1LeftLine
    global prevX2Leftline
    
    global prevMRightLine
    global prevBRightLine
    global prevX1RightLine
    global prevX2Rightline

    totalPosSlope = 0
    totalNegSlope = 0
    posSlopeLines = []
    negSlopeLines = []

    for line in lines:
        for x1,y1,x2,y2 in line:
            m = (y2 - y1) / (x2 - x1)
            if m > 0:
                totalPosSlope = totalPosSlope + m
                posSlopeLines.append(line)
            else:
                totalNegSlope = totalNegSlope + m
                negSlopeLines.append(line)

    maxY = img.shape[0]
    minY = 330
    
    # We will be discarding lines that are off/different from the previous drawn line.
    # This threshold determines the difference under which we will discard lines.
    # The reason why we need to do this are:
    #    1.  Drawn lane lines should not differ greatly from frame to frame.  They should be
    #        gradual and progressive.
    #    2.  There could be lines/other objects on the image that are actually lines on the road
    #        but aren't part of the lane.  This could happen in the real world where lines are
    #        drawn on the road in error.  Alternatively, this could be a bug in our lane detection
    #        algorithm (this is hard to debug currently as I don't have the frame of the video where
    #        this is occurring.  Either way, let's discard these lines and draw the previous line 
    #        per reason 1 above.    
    threshold = 0.01
    
    numPosSlopeLines = len(posSlopeLines)            
    if numPosSlopeLines > 0:
        
        # Calculate the average slope of the right lines
        mRightLine = totalPosSlope / numPosSlopeLines
        
        if prevMRightLine is None or abs(mRightLine - prevMRightLine) < threshold:
            # Calculate and draw the new right line 
            prevMRightLine = mRightLine;
            rightLine = posSlopeLines[0]
            x = rightLine[0][0]
            y = rightLine[0][1]
            prevBRightLine = y - prevMRightLine * x
            prevX1RightLine = (maxY - prevBRightLine) / prevMRightLine
            prevX2Rightline = (minY - prevBRightLine) / prevMRightLine        
            cv2.line(img, (int(prevX1RightLine), maxY), (int(prevX2Rightline), minY), color, thickness)        
        else:
            # Draw the previous right line
            if (prevX1RightLine is not None) and (prevX2Rightline is not None):
                cv2.line(img, (int(prevX1RightLine), maxY), (int(prevX2Rightline), minY), color, thickness)        
            
    else:
        # Draw the previous right line
        if (prevX1RightLine is not None) and (prevX2Rightline is not None):
            cv2.line(img, (int(prevX1RightLine), maxY), (int(prevX2Rightline), minY), color, thickness)        

    numNegSlopeLines = len(negSlopeLines)            
    if numNegSlopeLines > 0:                   
             
        # Calculate the average slope of the left lines        
        mLeftLine = totalNegSlope / numNegSlopeLines
        
        if prevMLeftLine is None or abs(mLeftLine - prevMLeftLine) < threshold:
            # Calculate and draw the new left line 
            prevMLeftLine = mLeftLine;
            leftLine = negSlopeLines[0]
            x = leftLine[0][0]
            y = leftLine[0][1]
            prevBLeftLine = y - prevMLeftLine * x
            prevX1LeftLine = (maxY - prevBLeftLine) / prevMLeftLine
            prevX2Leftline = (minY - prevBLeftLine) / prevMLeftLine        
            cv2.line(img, (int(prevX1LeftLine), maxY), (int(prevX2Leftline), minY), color, thickness)
        else:
            # Draw the previous left line
            if (prevX1LeftLine is not None) and (prevX2Leftline is not None):
                cv2.line(img, (int(prevX1LeftLine), maxY), (int(prevX2Leftline), minY), color, thickness)        

    else:
        # Draw the previous left line
        if (prevX1LeftLine is not None) and (prevX2Leftline is not None):
            cv2.line(img, (int(prevX1LeftLine), maxY), (int(prevX2Leftline), minY), color, thickness)        

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)

    gray = grayscale(image)

    blur_gray = gaussian_blur(gray, 5)

    edges = canny(blur_gray, 50, 150)

    image_shape = image.shape
    vertices = np.array([[(0,image_shape[0]),(450, 330), (490, 330), (image_shape[1],image_shape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices);

    rho = 2 						# distance resolution in pixels of the Hough grid
    theta = np.pi/180 				# angular resolution in radians of the Hough grid

    threshold = 15     				# minimum number of votes (intersections in Hough grid cell)
                                    # i.e. at least 15 points in image space need to be associated
                                    # with each line segment

    min_line_length = 48 			# minimum number of pixels making up a line
    max_line_gap = 20    			# maximum gap in pixels between connectable line segments

    line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    image_with_lines = weighted_img(line_img, image, 0.8, 1, 0)
    return image_with_lines

for file in os.listdir("test_images/"):
    if file.endswith("jpg"):
        image = mpimg.imread("test_images/" + file)
        # Printing out some stats and plotting
        # print('This image is:', type(image), 'with dimensions:', image.shape)
        processed_image = process_image(image);
        #plt.imshow(processed_image)  # Call as plt.imshow(gray, cmap='gray') to show a grayscaled image
        #plt.show()
        mpimg.imsave("test_images/withDrawnLaneLines_" + file, processed_image)
 
