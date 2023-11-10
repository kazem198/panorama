import cv2
import numpy as np
import os

output = "outPut"
# use stitcher

# for folder in myFolders:
#     path = mainFolders + '/' + folder
#     images = []
#     myList = os.listdir(path)
#     for imgN in myList:
#         curImg = cv2.imread(f'{path}/{imgN}')
#         curImg = cv2.resize(curImg, (0, 0), None, 0.2, 0.2)
#         images.append(curImg)

#     stitcher = cv2.Stitcher.create()
#     status, result = stitcher.stitch(images)
#     print(status)
#     if (status == cv2.Stitcher_OK):
#         print("panorama generated")
#         cv2.imshow(folder, result)
#     else:
#         print("panorama generated unsuccessful")

orb = cv2.ORB_create()
bf = cv2.BFMatcher()


def setPanorama(img2, img1, i):

    img_1 = cv2.imread(img1)
    img_2 = cv2.imread(img2)
    img1_gary = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img2_gary = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(img1_gary, None)
    kp2, des2 = orb.detectAndCompute(img2_gary, None)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m in matches:
        if (m[0].distance < 0.5*m[1].distance):
            good.append(m)
    matches = np.asarray(good)

    if (len(matches[:, 0]) >= 4):

        src = np.float32(
            [kp1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        dst = np.float32(
            [kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    else:
        raise AssertionError('Can’t find enough keypoints.')

    dst = cv2.warpPerspective(
        # wraped image
        img_1, H, ((img_1.shape[1] + img_2.shape[1]), img_2.shape[0]))
    dst[0:img_2.shape[0], 0:img_2.shape[1]] = img_2  # stitched image

    pathOutPut = f'{output}/output{i}.jpg'
    cv2.imwrite(pathOutPut, dst)
    return pathOutPut


mainFolders = "images"
myFolders = os.listdir(mainFolders)

# setPanorama("output0.jpg", "output1.jpg", 2)

for folder in myFolders:
    path = mainFolders + '/' + folder

    myList = os.listdir(path)

    outputList = []
    pathoutput = setPanorama(path+"/"+myList[0], path+"/" + myList[1], 1)
    outputList.append(pathoutput)

    for i in range(2, len(myList), 1):

        pathoutput = setPanorama(pathoutput, path+"/"+myList[i], i)
        outputList.append(pathoutput)

    for i in range(0, len(outputList)-1, 1):
        pathoutput = setPanorama(outputList[i], outputList[i+1], i+15)
        print(outputList)


if cv2.waitKey(0) & 0xFF == ord("q"):
    cv2.destroyAllWindows()
