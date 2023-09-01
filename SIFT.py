import numpy
import cv2
import imutils

i=1
while i<=50:
    if i < 10:
        image = cv2.imread('dataset/00' + str(i) + '.jpg')
    else:
        image = cv2.imread('dataset/0' + str(i) + '.jpg')
    
    image = imutils.resize(image, width=700)
    img = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    keypoints, desciptors = sift.detectAndCompute(gray, None)
    
    img = cv2.drawKeypoints(image = image, outImage=img, keypoints=keypoints,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(0,0,0))
    
    if i<10:
        cv2.imwrite('SIFT/SIFT00' + str(i) + '.jpg', img)
    else:
        cv2.imwrite('SIFT/SIFT0' + str(i) + '.jpg', img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    i+=1
        