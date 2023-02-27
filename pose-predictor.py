import cv2
import mediapipe as mp
import time
import csv
import pickle

import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


# Load the saved classifier model from the binary file
with open('classifier.pickle', 'rb') as f:
    classifier = pickle.load(f)


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
        image = cv2.putText(image, "Lag en pose", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, 1)
        

        # TODO: Lage samme array som i fila
        tallArr = []
        for obj in results.pose_landmarks.landmark:
            tallArr.extend(
                np.round([obj.x, obj.y, obj.z, obj.visibility], 2))
        # You can now use the "classifier" object to make predictions
        y_pred = classifier.predict([tallArr])

        text = y_pred[0]
        image = cv2.putText(
            image, text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, 1)
        cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
