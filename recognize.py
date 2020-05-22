import cv2
import numpy as np 

faceDetect = cv2.CascadeClassifier("D:\\python_exmple\\face_recognition\\haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()

rec.read("D:\\python_exmple\\face_recognition\\recognizer\\trainingData.yml")
id = 0
font = cv2.FORMATTER_FMT_CSV

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 15)
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        id, conf = rec.predict(gray[y:y+h, x:x+w])

        # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        # id, conf = rec.predict(gray[y:y+h, x:x+w])


        if id == 1:
            id="Robel"  
        if id == 2:
            id="Girmay"
        if id == 2:
            id="Elias"
        if id == 2:
            id="Biruk"  
        if id == 2:
            id="Habtsh"
        if id == 2:
            id="Selam"
        if id == 2:
            id="Awet"  

        cv2.putText(img, str(id), (x, y+h), font, 1, (250, 200, 250))

    cv2.imshow("Face", img)
    if cv2.waitKey(1)==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()        












