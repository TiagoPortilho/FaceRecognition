import cv2
import numpy as np
import os

#OpenCV facial detection pre-trained model using Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)
faces_folder = "faces"
faces_list = []

for filename in os.listdir(faces_folder):
    if filename.endswith('.jpg'):
        # Create the path for the file
        image_path = os.path.join(faces_folder, filename)

        # Read the image and add to list
        image = cv2.imread(image_path)
        if image is not None:
            faces_list.append(image)

if not faces_list:
    print("Error: No reference image is loaded. Please, start record_face.py")
    input(print("Press ENTER to continue"))
    exit()

reference_gray = cv2.cvtColor(faces_list[0], cv2.COLOR_BGR2GRAY)

dimensions = reference_gray.shape
print("Reference image dimensions:", dimensions)
print("Press Q to leave")

while True:
    ret, frame = video_capture.read()

    # Converts the image to grayscale, necessary for facial detection with Haar in OpenCV
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(35, 35))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_roi = gray[y:y + h, x:x + w]
        face_roi_resized = cv2.resize(face_roi, dimensions)

        for face in faces_list:
            reference_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            reference_gray = cv2.resize(reference_gray, dimensions)

            hist_reference = cv2.calcHist([reference_gray], [0], None, [256], [0, 256])
            hist_face = cv2.calcHist([face_roi_resized], [0], None, [256], [0, 256])

            cv2.normalize(hist_reference, hist_reference, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist_face, hist_face, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            correlation = cv2.compareHist(hist_reference, hist_face, cv2.HISTCMP_CORREL)

            if correlation > 0.8:  # Adjust if needed
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Face Recognized", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()