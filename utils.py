import cv2
import numpy as np

def rotate_and_scale(img, angle, scale): 
    # Taking image height and width 
    height, width = img.shape[0], img.shape[1] 
  
    # Computing the centre x,y coordinates 
    # of an image 
    centreY, centreX = height//2, width//2
  
    # Computing 2D rotation Matrix to rotate an image 
    rotationMatrix = cv2.getRotationMatrix2D((centreY, centreX), angle, scale) 
  
    # Now will take out sin and cos values from rotationMatrix 
    # Also used numpy absolute function to make positive value 
    cosofRotationMatrix = np.abs(rotationMatrix[0][0]) 
    sinofRotationMatrix = np.abs(rotationMatrix[0][1]) 
  
    # Now will compute new height & width of 
    # an image so that we can use it in 
    # warpAffine function to prevent cropping of image sides 
    new_height = int((height * sinofRotationMatrix) +
                         (width * cosofRotationMatrix)) 
    new_width = int((height * cosofRotationMatrix) +
                        (width * sinofRotationMatrix)) 
  
    # After computing the new height & width of an image 
    # we also need to update the values of rotation matrix 
    rotationMatrix[0][2] += (new_width/2) - centreX 
    rotationMatrix[1][2] += (new_height/2) - centreY 
  
    # Now, we will perform actual image rotation 
    if len(img.shape) == 3:
        borderValue = (255,255,255)
    else:
        borderValue = 1
    rotatingimage = cv2.warpAffine( 
        img, rotationMatrix, (new_width, new_height), borderValue=borderValue) 
  
    return rotatingimage 