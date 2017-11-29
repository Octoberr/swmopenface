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
# predictdatabase = os.path.join(classifierDir, 'classifier.pkl')  # 人脸数据库
imgdir = os.path.join(fileDir, '..', 'unknown')
train = os.path.join(fileDir, '..', 'train', 'songwangmeng')
huge = os.path.join(imgdir, 'huge')
jiesen = os.path.join(imgdir, 'jiesen')


def selectrep():
    con = sqlite3.connect('new')
    res = con.execute("SELECT * FROM bestwise").fetchall()
    return res


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


def getdatarep():
    X = []  # 存储人脸特征
    y = []  # 存储人脸的名字
    # im_names = os.listdir(train)
    # for imgname in im_names:
    #     imgpath =os.path.join(train, imgname)
    #     imgrep = getrep(imgpath)
    #     repname = 'swm'
    #     X.append(imgrep)
    #     y.append(repname)
    #     print 'get success!'
    res = selectrep()
    for element in res:
        rep = convert_array(element[1])
        X.append(rep)
        y.append(element[0])
    X = np.vstack(X)
    y = np.array(y)
    # GridSearchSvm
    param_grid = [
        {'C': [1, 10, 100, 1000],
         'kernel': ['linear']},
        {'C': [1, 10, 100, 1000],
         'gamma': [0.001, 0.0001],
         'kernel': ['rbf']}
    ]
    svm = GridSearchCV(SVC(C=1, probability=True), param_grid, cv=5).fit(X, y)
    return svm


def getrep(imgpath):
    bgrImg = cv2.imread(imgpath)
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    bb = align.getLargestFaceBoundingBox(rgbImg)
    alignedFace = align.align(96, rgbImg, bb,
                              landmarkIndices=landmarkIndices)
    rep = net.forward(alignedFace)
    return rep


# 获取人脸的处理
def getRep(img):
    bgrImg = cv2.imread(img)
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    bbs = align.getAllFaceBoundingBoxes(rgbImg)
    reps = []
    for bb in bbs:
        facelandmarks = align.findLandmarks(rgbImg, bb)
        alignedFace = align.align(96, rgbImg, bb, facelandmarks, landmarkIndices=landmarkIndices)
        rep = net.forward(alignedFace)
        reps.append((bb.center().x, rep))
    sreps = sorted(reps, key=lambda x: x[0])
    return sreps


if __name__ == '__main__':
    svm = getdatarep()
    # img = os.path.join(huge, 'huge10.jpg')
    img = 'swm.jpg'  # 多个人
    reps = getRep(img)
    data = []
    for r in reps:
        rep = r[1].reshape(1, -1)
        bbx = r[0]
        prediction = svm.predict_proba(rep)[0]
        people = svm.predict(rep)[0]
        data.append((np.max(prediction), people))
    print(data)




