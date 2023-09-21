import mediapipe as mp
import numpy as np
import cv2
from handle_pose import *
class detect:
    def __init__(self,model):
        self.model=model
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.mpDraw = mp.solutions.drawing_utils
        self.handle_mediapipe=handle_pose
    def detect_physical(self,image):
        results = self.pose.process(image)
        if results.pose_landmarks:
            lm_list = np.array(self.handle_mediapipe.make_landmark_timestep(results))
            action=np.argmax(self.model.predict(lm_list.reshape(-1,132,)))
            return action
        return False

    
