from pyexpat import model
import cv2
from grpc import server
import mediapipe as mp
import numpy as np
import tensorflow as tf
import threading
import time
import socket
from detect import *
from handle_pose import *
from handelPushUp import *
import pyshine as ps
from flask import Flask, render_template, render_template_string, Response

handlepose=handle_pose()
#***********************************SETUP SERVER*****************************************
HOST = socket.gethostbyname(socket.gethostname())
PORT = 80

#*****************************************************************************************************

app = Flask(__name__)
@app.route('/')
def index():
    """Video streaming"""
    #return render_template('index.html')
    return render_template_string('''<html>
<head>
</head>
<body>
    <div>
         <center><canvas id="canvas" "></canvas> </center>
    </div>

<script >
    var ctx = document.getElementById("canvas").getContext('2d');
    var img = new Image();
    img.src = "{{ url_for('video_feed') }}";

    // need only for static image
    //img.onload = function(){   
    //    ctx.drawImage(img, 0, 0);
    //};

    // need only for animated image
    function refreshCanvas(){
        ctx.drawImage(img, 0, 0);
    };
    window.setInterval("refreshCanvas()", 50);

</script>

</body>
</html>''')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                mimetype='multipart/x-mixed-replace; boundary=frame')

#*****************************************************************************************************

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
        global feature,option_feature
        data = conn.recv(4096).decode()
        data=data.split(",")
        feature = data[0]
        option_feature = data[1]

        if not data: break
        print(data,type(data))
        conn.close()
        break
#SERVER START
server = threading.Thread(target=_start)
server.start()

#*************************************************************************************

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)

#*************************************************************************************
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
models = tf.keras.models.load_model("model-test.h5")
modelyoga = tf.keras.models.load_model("modelyoga.h5")

classes=['anantasana','bakasana','balasana','bhekasana']
classesYoga=['1','2','3','4']

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
K=setK(len(classes))
orderOfAction=0
Time=5
stop_threed=False
t1.start()
#********************************SETUP PUSH UP******************************************#
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
def draw_result_detect_yoga_on_image(label,Time,img):
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
def gen():
    counter = 0 
    stage = None
    modeldetect=detect(models)
    modeldetectyoga=detect(modelyoga)
    feature="2"
    option_feature="squats"
    while (cap. isOpened()):
        ret,img = cap.read()
        results=pose.process(img)
        if(feature=="1"):
            # nhan dien dong tac
            action=modeldetect.detect_physical(img)
            if detect_movements(action,K)==True:
                img=handlepose.draw_landmark_on_image(results=results,img=img)
                if(Time==0):
                    #cout down
                    K[action]=True
                    orderOfAction=orderOfAction+1
                    Time=60
                img = draw_result_detect_action_on_image(label=classes[action],Time=Time,img= img)
            else:
                note=str("dong tac dang thuc hien: "+classes[action] +" dong tac hien tai la: " + str(orderOfAction))
                img =draw_result_detect_action_on_image(label=note,Time=Time,img=img)
                Time=60
        if(feature=="2"):
            if(results.pose_landmarks):  
                landmarks =results.pose_landmarks.landmark
                # Get coordinates
                shoulder_left = [landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].y]
                shoulder_right = [landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].y]

                RIGHT_HIP = [landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].y]
                RIGHT_KNEE = [landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value].y]
                RIGHT_ANKLE = [landmarks[mpPose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mpPose.PoseLandmark.RIGHT_ANKLE.value].y]        

                # Calculate angle
                if(option_feature=="dumbbellcurl"):
                    angle = calculate_angle(shoulder_left, elbow, wrist)
                if(option_feature=="squats"):
                    angle = calculate_angle(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
                if(option_feature=="crunch"):
                    angle = calculate_angle(shoulder_right, RIGHT_HIP, RIGHT_KNEE)
                # Visualize angle
                if(option_feature!=""):
                    if angle > 160:
                            stage = "down"
                    if angle < 30 and stage =='down':
                            stage="up"
                            counter +=1
                if(option_feature=="dumbbellcurl"):
                    img= draw_result_pushUp_on_image(counter,stage,angle,elbow,img)
                if(option_feature=="squats"):
                    img= draw_result_pushUp_on_image(counter,stage,angle,RIGHT_KNEE,img)
                if(option_feature=="crunch"):
                    img= draw_result_pushUp_on_image(counter,stage,angle,RIGHT_HIP,img)
                
                # Render detections
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS,
                                        mpDraw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mpDraw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                        )
        if(feature=="3"):
            yogaAction=0
            action=modeldetectyoga.detect_physical(img)
            img=handlepose.draw_landmark_on_image(results=results,img=img)
            print(action)
            if(action==yogaAction):
                img = draw_result_detect_yoga_on_image(label=classesYoga[action],Time=Time,img= img)
            else:
                Time=60
                img = draw_result_detect_yoga_on_image(label=classesYoga[action],Time=Time,img= img)
                
        cv2.imwrite('t.jpg', img)
        yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + open('t.jpg', 'rb').read() + b'\r\n')
        if cv2.waitKey(1) == ord('q'):
            stop_threed=True
            t1.join()
            break
app.run(host="192.168.1.14")
SERVER.close()
cap.release()
cv2.destroyAllWindows()