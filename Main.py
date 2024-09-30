import cv2
import numpy as np

#OpenCV facial detection pre-trained model using Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)
me = cv2.imread("face_1.jpg")
if me is None:
    print("Erro: a imagem de referência não foi encontrada ou não pôde ser carregada.")


#The first parameter is your image
reference_gray = cv2.cvtColor(me, cv2.COLOR_BGR2GRAY)
reference_gray = cv2.resize(reference_gray, (300, 300))
print("Dimensões da imagem de referência:", reference_gray.shape)

face_id = 1

while True:
    ret, frame = video_capture.read()

    #Converts the image to grayscale, necessary for facial detection with Haar in OpenCV
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(35, 35))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_roi = gray[y:y + h, x:x + w]
        face_roi_resized = cv2.resize(face_roi, (300,300))

        hist_reference = cv2.calcHist([reference_gray], [0], None, [256], [0, 256])
        hist_face = cv2.calcHist([face_roi_resized], [0], None, [256], [0, 256])

        cv2.normalize(hist_reference, hist_reference, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_face, hist_face, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        correlation = cv2.compareHist(hist_reference, hist_face, cv2.HISTCMP_CORREL)

        # Define um limiar para o reconhecimento
        if correlation > 0.7:  # Ajuste esse valor conforme necessário
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Rosto reconhecido!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()