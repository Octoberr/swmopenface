# coding: utf-8
import os
import cv2
import openface
import numpy as np

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
predict = os.path.join(dlibModelDir, 'shape_predictor_68_face_landmarks.dat')
torchmodel = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')
align = openface.AlignDlib(predict)
net = openface.TorchNeuralNet(torchmodel)
landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE


def getRep(imgPath):
    bgrImg = cv2.imread(imgPath)
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    bb = align.getLargestFaceBoundingBox(rgbImg)
    alignedFace = align.align(96, rgbImg, bb,
                              landmarkIndices=landmarkIndices)
    # rep = net.forward(alignedFace)
    return alignedFace


if __name__ == '__main__':
    # huge
    # image1 = "huge1.jpg"
    image2 = "huge2.jpg"
    # swm
    # image3 = "swm.jpg"
    align2 = getRep(image2)
    print(align2)