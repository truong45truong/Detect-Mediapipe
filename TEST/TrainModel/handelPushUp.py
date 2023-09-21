import numpy as np
import cv2
class handlePushUp:
    def __init__(self) -> None:
        pass
def draw_result_pushUp_on_image(reps,stage,angle,elbow,img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.rectangle(img, (0,0), (225,73), (245,117,16), -1)
    cv2.putText(img, str(angle), 
                            tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
            
    # Rep data
    cv2.putText(img, "REPS",
                    (15,12),
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
    cv2.putText(img, str(reps),
                    (10,60),
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)            
    # Stage data
    cv2.putText(img, 'STAGE',
                    (65,12),
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)  
    cv2.putText(img, str(stage),
                    (65,60),
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)                  
    return img

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle >180.0:
        angle = 360-angle
    return angle 

