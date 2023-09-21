import mediapipe as mp
import numpy as np
import cv2
class handle_pose:
    def __init__(self):
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.mpDraw = mp.solutions.drawing_utils
    def make_landmark_timestep(results):
        c_lm = []
        for id, lm in enumerate(results.pose_landmarks.landmark):
            c_lm.append(lm.x)
            c_lm.append(lm.y)
            c_lm.append(lm.z)
            c_lm.append(lm.visibility)
        return c_lm
    def draw_landmark_on_image(self,results,img):
        if(results.pose_landmarks):
            self.mpDraw.draw_landmarks(img,results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return img
    def draw_result_detect_action_on_image(label,Time,img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 30)
        fontScale = 1
        fontColor = (0, 255, 0)
        thickness = 2
        lineType = 2
        cv2.putText(img, label,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
        cv2.putText(img,str(Time),
                    (1800,30),
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)      
        return img
