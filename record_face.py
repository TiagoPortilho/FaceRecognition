from time import sleep
import cv2
import os

face_id = 0

def recordFace(faces_id):
    video_capture = cv2.VideoCapture(0)
    files_path = "faces"
    os.makedirs(files_path, exist_ok=True)
    faces_idmax = faces_id + 10
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    while True:
        ret, frame = video_capture.read()

        # Converts the image to grayscale, necessary for facial detection with Haar in OpenCV
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(35, 35))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            face_roi = gray[y:y + h, x:x + w]
            face_roi_resized = cv2.resize(face_roi, (300, 300))



            if faces_id < faces_idmax + 1:
                with open(os.path.join(files_path, f'face_{faces_id}.jpg'), 'w') as jpg:
                    cv2.imwrite(os.path.join(files_path, f'face_{faces_id}.jpg'), face_roi)
                print(faces_id)
                faces_id += 1
            elif faces_id > faces_idmax:
                video_capture.release()
                cv2.destroyAllWindows()
                return faces_id

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

input(print("Look right and press ENTER"))
face_id = recordFace(face_id)

