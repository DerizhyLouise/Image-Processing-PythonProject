import cv2
import numpy as np

i = 1
while i<=50:
    if i < 10:
        img1 = cv2.imread('dataset/00' + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE)
        if i+1<10:
            img2 = cv2.imread('dataset/00' + str(i+1) + '.jpg', cv2.IMREAD_GRAYSCALE)
        elif i+1>50:
            img2 = cv2.imread('dataset/001.jpg', cv2.IMREAD_GRAYSCALE)
        else:
            img2 = cv2.imread('dataset/0' + str(i+1) + '.jpg', cv2.IMREAD_GRAYSCALE)
    else:
        img1 = cv2.imread('dataset/0' + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE)
        if i+1>50:
            img2 = cv2.imread('dataset/001.jpg', cv2.IMREAD_GRAYSCALE)
        else:
            img2 = cv2.imread('dataset/0' + str(i+1) + '.jpg', cv2.IMREAD_GRAYSCALE)
            
        
            

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)

    res = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

    if i<10:
        cv2.imwrite('Matching/Matching00' + str(i) + '.jpg', res)
    else:
        cv2.imwrite('Matching/Matching0' + str(i) + '.jpg', res)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    i+=1