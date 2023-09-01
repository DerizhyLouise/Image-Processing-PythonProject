import cv2
import numpy as np

i=1
while i<=50:
    if i < 10:
        img = cv2.imread('dataset/00' + str(i) + '.jpg')
    else:
        img = cv2.imread('dataset/0' + str(i) + '.jpg')
         
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(gray,(3,3),0)

    #canny
    img_canny = cv2.Canny(img,100,200)

    #sobel
    img_sobelx = cv2.Sobel(img_gaussian,cv2.CV_8U,1,0,ksize=5)
    img_sobely = cv2.Sobel(img_gaussian,cv2.CV_8U,0,1,ksize=5)
    img_sobel = img_sobelx + img_sobely


    #prewitt
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
    img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
    img_prewitt = img_prewittx + img_prewitty

    # cv2.imshow("Original Image", img)
    # cv2.imshow("Sobel", img_sobel)
    # cv2.imshow("Prewitt", img_prewitt)

    if i<10:
        cv2.imwrite('Sobel/Sobel00' + str(i) + '.jpg', img_sobel)
        cv2.imwrite('Prewitt/Prewitt00'+ str(i) + '.jpg', img_prewitt)
    else:
        cv2.imwrite('Sobel/Sobel0' + str(i) + '.jpg', img_sobel)
        cv2.imwrite('Prewitt/Prewitt0'+ str(i) + '.jpg', img_prewitt)
        

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    i+=1
    
