# coding:utf-8
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
align = openface.AlignDlib(predict)
rgb = openface.data.Image('handp', 'handp', 'yangmi3andliyifeng.jpg').getRGB()
# outRgb = align.align(96, rgb, landmarkIndices=landmarkIndices, skipMulti=True)
# deepFunneled = "{}/{}.jpg".format(os.path.join(fileDir,'swm'), 'test1')
# shutil.copy(deepFunneled, "{}/{}.jpg".format(os.path.join(fileDir,'swm'), 'test1'))
# print deepFunneled
# outBgr = cv2.cvtColor(outRgb, cv2.COLOR_RGB2BGR)
# cv2.imwrite(imgName, outBgr)
allface = align.getAllFaceBoundingBoxes(rgb)
print allface
i = 0
for face in allface:
    bl = (face.left(), face.bottom())
    tr = (face.right(), face.top())
    print bl, type(bl)
    print tr, type(tr)
    i += 1
    facelandmarks = align.findLandmarks(rgb, face)
    outRGB = align.align(96, rgb, face, facelandmarks, landmarkIndices=landmarkIndices, skipMulti=True)
    outBgr = cv2.cvtColor(outRGB, cv2.COLOR_RGB2BGR)
    imgName = str(i) + '.png'
    cv2.imwrite(imgName, outBgr)