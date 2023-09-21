import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import os
import PIL 
import threading
import time
import socket as soc

cap = cv2.VideoCapture(0)
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils


HOST = "192.168.56.1"  # The server's hostname or IP address
PORT = 80  # The port used by the server

s= soc.socket(soc.AF_INET, soc.SOCK_STREAM)
s.connect((HOST, PORT))
#s.sendall(b"Hello, world")
#data = s.recv(1024)

classes={'anantasana':[1,0,0,0],'bakasana':[0,1,0,0],'balasana':[0,0,1,0],'bhekasana':[0,0,0,1]}
classes1=['anantasana','bakasana','balasana','bhekasana']
TRAIN_DATA='dataset/train-data'
TEST_DATA='dataset/test-data'
def read_File_Img(link):
    Img_list=[]
    for folder in os.listdir(link):
        folder_path=os.path.join(link,folder)
        list_filename_path=[]
        for filename in os.listdir(folder_path):
            filename_path=os.path.join(folder_path,filename)
            label=filename_path.split('\\')[1]
            img=np.array(PIL.Image.open(filename_path))
            list_filename_path.append((img,classes[label]))
        Img_list.extend(list_filename_path)
    return Img_list

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


def draw_class_on_image(label,Time, img):
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
lm_list=[]
img_to_server=None
def senddata():
    while True:
        time.sleep(1)
        print(img_to_server)
        s.sendall(img_to_server.encode('utf-8'))
t2=threading.Thread(target=senddata)
t2.start()

while (cap. isOpened()):
    ret,img = cap.read()
    img_to_server=img
    results = pose.process(img)
    if results.pose_landmarks:
        # Ghi nhận thông số khung xương
        lm = np.array(make_landmark_timestep(results))
        # Vẽ khung xương lên ảnh
        img = draw_landmark_on_image(mpDraw, results,img)
        # nhan dien dong tac
        lm_list=lm.reshape(-1,132,)
        action=0
        if detect_movements(action,K)==True:
            if(Time==0):
                #cout down
                K[action]=True
                orderOfAction=orderOfAction+1
                Time=60
            img = draw_class_on_image(classes1[action],Time, img)
        else:
            note="dong tac dang thuc hien: "+classes1[action] +" dong tac hien tai la: " + str(orderOfAction)
            img = draw_class_on_image(note,Time, img)
            Time=60
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        stop_threed=True
        t1.join()
        break

cap.release()
cv2.destroyAllWindows()