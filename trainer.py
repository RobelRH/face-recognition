import os
import cv2
import numpy as np 
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()

path = "D:/python_exmple/face_recognition/dataset"

def getimagewithid(path):
    imagePath = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    IDs = []
    for impath in imagePath:
        faceimg = Image.open(impath).convert('L')
        faceNp = np.array(faceimg, 'uint8') #user.1.1
        ID = int(os.path.split(impath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("training", faceNp)
        cv2.waitKey(10)
    return IDs, faces

Ids, faces = getimagewithid(path)
recognizer.train(faces, np.array(Ids))
recognizer.save('D:/python_exmple/face_recognition/recognizer/trainingData.yml')
cv2.destroyAllWindows()



