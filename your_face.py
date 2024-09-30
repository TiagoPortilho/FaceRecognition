import cv2

#OpenCV facial detection pre-trained model using Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)
face_id = 1

while True:
    ret, frame = video_capture.read()

    # Converts the image to grayscale, necessary for facial detection with Haar in OpenCV
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(35, 35))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_roi = gray[y:y + h, x:x + w]
        face_roi_resized = cv2.resize(face_roi, (300, 300))

        face_filename = f'face_{face_id}.jpg'
        cv2.imwrite(face_filename, face_roi)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
