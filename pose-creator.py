import cv2
import mediapipe as mp
import time
import csv

import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

start_time = time.time()
message = "Press space to start capture"
mode = "wait"
# poses = ["RaiseLeftArm", "RaiseRightArm", "RiseBothArms", "Testing"]
poses = ["Testing"]
poseNo = 0

isCapture = False
captureCount = 0
captureArray = []

cap = cv2.VideoCapture(0)
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        #print(results.pose_landmarks)
        # Flip the image horizontally for a selfie-view display.
        image = cv2.flip(image, 1)
        image = cv2.putText(image, message, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, 1)
        image = cv2.putText(
            image, poses[poseNo], (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, 1)
        cv2.imshow('MediaPipe Pose', image)

        elapsed_time = time.time() - start_time

        if mode == "wait" and cv2.waitKey(3) & 0xFF == 32:
            print('starting')
            mode = "getReady"
            start_time = time.time()+3

        if mode == "getReady" and start_time > time.time():
            message = "Start in " + str(round(elapsed_time, 1)) + " sec"

        if mode == "getReady" and start_time < time.time():
            mode = "capture"

        if mode == "capture":
            captureArray.append(results.pose_landmarks)
            message = "Capturing..." + str(round(elapsed_time, 1))

        # Write captured data to file
        if mode == "capture" and elapsed_time >= 3:
            message = "Press space to start capture"
            mode = "wait"
            with open("./data/" + poses[poseNo]+".csv", "w", newline='') as file:
                writer = csv.writer(file, delimiter=';')
                for arr in captureArray:
                    tallArr = []
                    for obj in arr.landmark:
                        tallArr.extend(
                            np.round([obj.x, obj.y, obj.z, obj.visibility], 2))
                    writer.writerow(tallArr)
            print('stopped')
            if (poseNo < len(poses)-1):
                poseNo += 1
                captureArray = []
            else:
                break
                                

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
