import mediapipe as mp
import cv2
import numpy as np
import math
import uuid
import os
import time
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER

# import pycaw.pycaw

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

devices = AudioUtilities.GetSpeakers()
interfaces =  devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interfaces, POINTER(IAudioEndpointVolume))

volume_range = volume.GetVolumeRange()

min_vol = volume_range[0]
max_vol = volume_range[1]

vol = 0
vol_bar = 400
vol_percent = 0
length = 0

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         
        image = cv2.flip(image, 1) # flip to mirror
        image.flags.writeable = False
        
        results = hands.process(image)
        image.flags.writeable = True
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        lmlist = []
        if results.multi_hand_landmarks:
            myHand = results.multi_hand_landmarks[0]
            for id, dot in enumerate(myHand.landmark):
                h, w, c = image.shape
                cx, cy = int(dot.x * w), int(dot.y * h)
                lmlist.append([id, cx, cy])

        if len(lmlist) != 0:
            x1, y1 = lmlist[4][1], lmlist[4][2]
            x2, y2 = lmlist[8][1], lmlist[8][2]

            cv2.line(image, (x1, y1), (x2, y2), (255,200,0), 3)
            length = math.hypot(x2-x1, y2-y1)
            
        vol = np.interp(length, [60, 330], [min_vol, max_vol])
        vol_bar = np.interp(length, [60,330], [400, 150])
        vol_percent = np.interp(length, [50, 300], [0,100])
        volume.SetMasterVolumeLevel(vol, None)

        cv2.rectangle(image, (50,150), (85, 400), (255, 200, 0), 3)
        cv2.rectangle(image, (50, int(vol_bar)), (85, 400), (255,200,0), cv2.FILLED)
        cv2.putText(image, f'Volume:{int(vol_percent)}%', (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,215,0),2)

        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()