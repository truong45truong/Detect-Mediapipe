import cv2
from grpc import server
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import PIL 
import threading
import time
import socket

from TEST.TrainModel.handelPushUp import calculate_angle

#***********************************SETUP SERVER*****************************************
HOST = socket.gethostbyname(socket.gethostname())
PORT = 80

SERVER = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    SERVER.bind((HOST,PORT))
    print(f'* Running on http://{HOST}:{PORT}')
except socket.error as e:
    print(f'socket error: {e}')
    print('socket error: %s' %(e))

def _start():
    SERVER.listen()
    while True:
        conn, addr = SERVER.accept()
        print(conn,addr)
        thread = threading.Thread(target=_handle, args=(conn, addr))
        thread.start()
def _handle(conn,addr):
    while True:
        data = conn.recv(4096)
        if not data: break
        print(data.decode())
        conn.close()
        break
#SERVER START
server = threading.Thread(target=_start)
server.start()

#*************************************************************************************
cap = cv2.VideoCapture(0)



#*************************************************************************************
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
models = tf.keras.models.load_model("model-test.h5")


classes={'anantasana':[1,0,0,0],'bakasana':[0,1,0,0],'balasana':[0,0,1,0],'bhekasana':[0,0,0,1]}
classes1=['anantasana','bakasana','balasana','bhekasana']

def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return img


def draw_result_detect_action_on_image(label,Time, img):
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
def result_landmark(List_img):
    # Khởi tạo thư viện mediapipe
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    lm_list = []
    for i in List_img:
        # Nhận diện pose
        results = pose.process(i)
        if results.pose_landmarks:
            # Ghi nhận thông số khung xương
            lm = make_landmark_timestep(results)
            lm_list.append((lm,i[1]))
    return lm_list
def detect_movements(action,K):
    global orderOfAction
    if orderOfAction == action and K[action] == False:
        return True
    return False 
# set K[F,F,....,F]
def setK(lenK):
    K=[]
    for i in range(0,lenK):
        K.append(False)
    return K
#dem nguoc thoi gian cua dong tac thuc hien
def countTime():
    global Time
    global stop_threed
    while True:
        time.sleep(1)
        Time=Time-1
        print(Time)
        if stop_threed == True:
            break
# cho chay da luong
t1=threading.Thread(target=countTime)
K=setK(len(classes1))
orderOfAction=0
Time=5
stop_threed=False
t1.start()
#********************************SETUP PUSH UP*******************************************
def draw_result_pushUp_on_image(reps,stage):
    cv2.rectangle(img, (0,0), (225,73), (245,117,16), -1)
            
    # Rep data
    cv2.putText(img, 
                'REPS', (15,12), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0,0,0), 
                1, 
                cv2.LINE_AA)
    cv2.putText(img, str(reps), 
                (10,60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                2, 
                (255,255,255), 
                2, 
                cv2.LINE_AA)
            
    # Stage data
    cv2.putText(img, 
                'STAGE', 
                (65,12), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0,0,0), 
                1, 
                cv2.LINE_AA)
    cv2.putText(img, stage, 
                (60,60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                2, (255,255,255), 
                2, 
                cv2.LINE_AA)

counter = 0 
stage = None
data="1"
while (cap. isOpened()):
    ret,img = cap.read()
    results = pose.process(img)
    if results.pose_landmarks:
        # Ghi nhận thông số khung xương
        lm = np.array(make_landmark_timestep(results))
        # Vẽ khung xương lên ảnh
        img = draw_landmark_on_image(mpDraw, results,img)
        if(data=="1"):
        # nhan dien dong tac
            action=np.argmax(models.predict(lm.reshape(-1,132,)))
            if detect_movements(action,K)==True:
                if(Time==0):
                    #cout down
                    K[action]=True
                    orderOfAction=orderOfAction+1
                    Time=60
                img = draw_result_detect_action_on_image(classes1[action],Time, img)
            else:
                note="dong tac dang thuc hien: "+classes1[action] +" dong tac hien tai la: " + str(orderOfAction)
                img = draw_result_detect_action_on_image(note,Time, img)
                Time=60
        if(data=="2"):
            try:
                landmarks =results.pose_landmarks.landmark
                # Get coordinates
                shoulder = [landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].y]
                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)
                # Visualize angle
                cv2.putText(img, str(angle), 
                            tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                if angle > 160:
                    stage = "down"
                if angle < 30 and stage =='down':
                    stage="up"
                    counter +=1
                    print(counter)
            except:
                pass
            draw_result_pushUp_on_image(counter,stage)
            
            # Render detections
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS,
                                    mpDraw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mpDraw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )  
    cv2.imshow("img", img)
    if cv2.waitKey(1) == ord('q'):
        stop_threed=True
        t1.join()
        break
SERVER.close()
cap.release()
cv2.destroyAllWindows()