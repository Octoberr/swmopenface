# coding:utf-8
import os
import cv2
import openface
import numpy as np
import random
import copy
from sklearn.svm import SVC

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
predict = os.path.join(dlibModelDir, 'shape_predictor_68_face_landmarks.dat')
torchmodel = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')
align = openface.AlignDlib(predict)
net = openface.TorchNeuralNet(torchmodel)
landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE
allimgdir = os.path.join(fileDir, 'imgcompare')
sameperson = 0.16

def getRep(imgPath):
    bgrImg = cv2.imread(imgPath)
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    bb = align.getLargestFaceBoundingBox(rgbImg)
    alignedFace = align.align(96, rgbImg, bb,
                              landmarkIndices=landmarkIndices)
    rep = net.forward(alignedFace)
    return rep


def getalldataSVM(X, y):
    X = np.vstack(X)
    y = np.array(y)
    svm = SVC(C=1, kernel='linear', probability=True).fit(X, y)
    return svm


def getdatarep():
    imgs = list(openface.data.iterImgs(allimgdir))
    X = []   # 存储rep
    y = []  # 存储name
    random.shuffle(imgs)
    # 将特征存储在sqlite
    allimgdata = []
    for imgobject in imgs:
        persondata = {}
        # print imgobject, type(imgobject)
        imgname = imgobject.name
        persondata['imgname'] = imgname
        rgb = imgobject.getRGB()
        bbs = align.getAllFaceBoundingBoxes(rgb)  # 人脸的框框
        if len(bbs) == 0:
            print 'no face in the image'
            continue
        tmprep = []
        for bb in bbs:
            landmarks = align.findLandmarks(rgb, bb)
            alignedFace = align.align(96, rgb, bb, landmarks=landmarks, landmarkIndices=landmarkIndices)
            if alignedFace is None:
                print 'no feature in the face. on another way, unknown problem.'
                continue
            rep = net.forward(alignedFace)
            print"processdata: ", imgname
            tmprep.append(rep)
            X.append(rep)
            y.append(imgname)
        persondata['rep'] = tmprep
        allimgdata.append(persondata)
    return allimgdata, X, y


def getallthesametype(alldata, rep, name):
    shaperep = rep.reshape(1, -1)
    predictions = alldata.predict_proba(shaperep).ravel()
    sametypeindex = [i for i in range(len(predictions)) if predictions[i] > sameperson]
    print name
    print predictions
    print sametypeindex
    tmpperson = [name[b] for b in sametypeindex]
    print tmpperson
    return tmpperson





def GetTheSameTyepImg(allimgdata):
    alldatasvm = getalldataSVM(allimgdata[1], allimgdata[2])
    allsametypename = []
    for data in allimgdata[0]:
        # if len(allsametypename) == 0:
        #     # 存储第一张图片的所有特征
        #     allsametypename += [[repsvm] for repsvm in data['repfeature']]
        #     nametype.append([data['imgname']])
        # else:
        #     # 以后的图片对比特征列表，有则添加
        for rep in data['rep']:
            tmp = getallthesametype(alldatasvm, rep, allimgdata[2])
            allsametypename.append(tuple(tmp))
    return set(allsametypename)

    # while len(allimgdata) > 1:
    #     tmpsametype = []
    #     delindex = []
    #     for i in range(len(samecopy)):
    #         standardlist = getallthediff(allimgdata[0]['rep'], samecopy[i]['rep'])
    #         print(samecopy[i]['imgname'], "get diff with ", allimgdata[0]['imgname'], "and the diffress: ", standardlist)
    #         # print(standard)
    #         if min(standardlist) < 0.5:
    #             tmpsametype.append(samecopy[i]['imgname'])
    #             delindex.append(i)
    #     allsametype.append(tmpsametype)
    #     for deli in reversed(delindex):
    #         del (allimgdata[deli])
    #         del (samecopy[deli])
    # return allsametype


if __name__ == '__main__':
    alldata = getdatarep()
    sametyep = GetTheSameTyepImg(alldata)
    print sametyep
    # print(alldata)
    # # huge
    # image1 = "huge1.jpg"
    # image2 = "huge2.jpg"
    # # swm
    # # image3 = "swm.jpg"
    # diff = getRep(image1) - getRep(image2)
    # print("compare with same photo.")
    # # print("compare with different photo.")
    # print(diff)
    # print("  + Squared l2 distance between representations: {:0.3f}".format(np.dot(diff, diff)))