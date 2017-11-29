# coding:utf-8
import os
import openface
import cv2
import pickle
import numpy as np
import random
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import sqlite3
import io

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
classifierDir = os.path.join(fileDir, '..', 'generated-embeddings')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
predict = os.path.join(dlibModelDir, 'shape_predictor_68_face_landmarks.dat')
torchmodel = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')
align = openface.AlignDlib(predict)
net = openface.TorchNeuralNet(torchmodel)
landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE
# traindir = os.path.join(fileDir, '..', 'aligned-images')
traindir = os.path.join(fileDir, '..', 'unknown')
sqliteFile = os.path.join(fileDir, 'peopledb')


# 插入数据库
def insertIntoSqlite(imgname, rep):
    con = sqlite3.connect('new')
    print 'insert name:', imgname
    con.execute("CREATE TABLE IF NOT EXISTS bestwise(name TEXT, array BLOB)")
    strrep = adapt_array(rep)
    con.execute("INSERT INTO bestwise(name, array) VALUES (?, ?)", (imgname, strrep))
    con.commit()
    con.close()


def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


def getdatarep():
    imgs = list(openface.data.iterImgs(traindir))
    random.shuffle(imgs)
    # 将特征存储在sqlite
    for imgobject in imgs:
        # print imgobject, type(imgobject)
        imgname = imgobject.cls
        print imgname
        rgb = imgobject.getRGB()
        bb = align.getLargestFaceBoundingBox(rgb)  # 人脸的框框
        if bb is None:
            continue
        landmarks = align.findLandmarks(rgb, bb)
        alignedFace = align.align(96, rgb, bb, landmarks=landmarks, landmarkIndices=landmarkIndices)
        if alignedFace is None:
            continue
        rep = net.forward(alignedFace)
        # strrep = adapt_array(rep)
        # newrep = convert_array(strrep)
        # print(strrep, type(strrep))
        # print(newrep, type(newrep))
        insertIntoSqlite(imgname, rep)
        print("insert databse success!!!!@@@@swm")


if __name__ == '__main__':
    getdatarep()

