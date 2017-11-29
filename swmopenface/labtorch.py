# coding:utf-8
"""
creat by swm
"""

import openface
import os
import shutil
import cv2
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
predict = os.path.join(dlibModelDir, 'shape_predictor_68_face_landmarks.dat')
landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE
torchmodel = os.path.join(openfaceModelDir, 'nn4.v2.t7')

align = openface.AlignDlib(predict)
rgb = openface.data.Image('swm', 'swm', 'swm.png').getRGB()
with openface.TorchNeuralNet(model=torchmodel) as net:
    print net.forward(rgb) # 提取一张图片的特征
