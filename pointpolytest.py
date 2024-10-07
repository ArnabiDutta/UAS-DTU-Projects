import cv2 as cv
import numpy as np


def triangles(img):
    grey=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # blurred=cv.GaussianBlur(grey,(7,7),0)
    edges=cv.Canny(grey,50,150)

    contours,_=cv.findContours(edges,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    centroids=[]
    for contour in contours:
        area=cv.contourArea(contour)
        approx=cv.approxPolyDP(contour,0.04 * cv.arcLength(contour, True), True)
        if len(approx)==3 and area>500:
            cv.drawContours(img,contour,-1,(0,0,0),5)
            M=cv.moments(edges)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv.circle(img, (cX, cY), 5, (255, 255, 255), -1)
            centroids.append({'centre':(cX,cY)})
            # triangle=np.array(triangle)

    print(len(centroids))
    return centroids

def centre_in_region(triangle,ext_contour):
    for point in triangle:
        if cv.pointPolygonTest(ext_contour,triangle['centre'],False)>=0:
        # if cv.pointPolygonTest(ext_contour,(point),False)>=0:
        
            return True
    return False    

