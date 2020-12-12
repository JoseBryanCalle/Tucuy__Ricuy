import cv2
import os
import numpy as np

rutaData = "C:/Users/ALEX/Desktop/Aldair Ase"
Lista = os.listdir(rutaData)

print('Lista de personas: ', Lista)
labels = []
facesData = []
label = 0
for nameDir in Lista:
    personPath = rutaData + '/' + nameDir
    print('Leyendo las imágenes')
    for fileName in os.listdir(personPath):
       
        labels.append(label)
        facesData.append(cv2.imread(personPath+'/'+fileName,0))
        image = cv2.imread(personPath+'/'+fileName,0)
        cv2.imshow('image',image)
        cv2.waitKey(10)
    label = label + 1

face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer.train(facesData,np.array(labels))
face_recognizer.write('modeloEigenFace2.xml')
