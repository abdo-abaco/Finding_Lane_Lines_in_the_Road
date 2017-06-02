#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
%matplotlib inline

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
#imageio.plugins.ffmpeg.download()

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """

    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    nRight = 0
    sumRxy = 0
    sumRx = 0
    sumRy = 0
    sumRx2 = 0
    
    nLeft = 0
    sumLxy = 0
    sumLx = 0
    sumLy = 0
    sumLx2 = 0
    

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            xAve = (x1+x2)/2
            
            # Right line vs Left Line : Linear Regression Y = mX + b
            if xAve > 480:
                print(x1, y1, x2, y2)
                #Linear Regression statistics
                nRight = nRight + 1
                sumRx = sumRx + x1 + x2
                sumRy = sumRy + y1 + y2
                sumRxy = sumRxy + x1*y1 + x2*y2
                sumRx2 = sumRx2 + x1*x1 + x2*x2
            else:
                #linear Regression statistics
                nLeft = nLeft + 1
                sumLx = sumLx + x1 + x2
                sumLy = sumLy + y1 + y2
                sumLxy = sumLxy + x1*y1 + x2*y2
                sumLx2 = sumLx2 + x1*x1 + x2*x2
                

    # calculating slope , m, from statistics
    mR = (2*nRight*sumRxy - sumRx*sumRy)/(2*nRight*sumRx2 - sumRx*sumRx)
    mL = (2*nLeft*sumLxy - sumLx*sumLy)/(2*nLeft*sumLx2 - sumLx*sumLx)
    
    # calculating y intercept, b, from statistics
    bL = (1/2/nLeft)*(sumLy - mL * sumLx)
    bR = (1/2/nRight)*(sumRy - mR * sumRx)
    
    # Left line parameters computed
    yLtop = (int)(img.shape[0]*0.6)
    yLbot = (int)(img.shape[0])
    yLdelta = yLbot - yLtop
    xLdelta = (int)(yLdelta*mL)
    xLtop = (int)((yLtop - bL)/mL)
    xLbot = (int)((yLbot-bL)/mL)
    #cv2.line(img, (xLtop, yLtop), (xLbot,yLbot ), color, thickness)
    
    yRtop = (int)(img.shape[0]*0.6)
    yRbot = (int)(img.shape[0])
    yRdelta = yRbot - yRtop
    xRdelta = (int)(yRdelta*mR)
    xRtop = (int)((yRtop - bR)/mR)
    xRbot = (int)((yRbot-bR)/mR)
    #cv2.line(img, (xRtop, yRtop), (xRbot,yRbot ), color, thickness)




# NOTE: The output you return should be a color image (3 channel) for processing video below
# TODO: put your pipeline here,
# you should return the final output (image where lines are drawn on lanes)

image = plt.imread('test_images/solidWhiteRight.jpg')

#converting to gray
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#gaussian blur
blur = cv2.GaussianBlur(gray, (3, 3), 0)

#gradient with thresholds
canny = cv2.Canny(blur, 70, 170)


# Next we'll create a masked edges image using cv2.fillPoly()
mask = np.zeros_like(canny)   
ignore_mask_color = 255

#Defining a four-sided polygon to mask
imshape = image.shape
vertices = np.array([[(0,imshape[0]),(imshape[1]*0.45, imshape[0]*0.65), (imshape[1]*0.55, imshape[0]*0.65), (imshape[1],imshape[0])]], dtype=np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(canny, mask)

#line detection
lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 10, np.array([]), 10, 5)
line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
draw_lines(line_img, lines)
image_lines = cv2.addWeighted(image, .9, line_img, 1, 0)



#printing out some stats and plotting
plt.imshow(line_img, cmap='gray')  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
plt.imsave('line_img.jpg', line_img, cmap='gray')




white_output = 'test_videos/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
#clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidYellowLeft.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)
