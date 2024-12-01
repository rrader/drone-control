import cv2
from math import pi
import numpy as np


def find_railways(img):
    # чорні обʼєкти
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    f = cv2.inRange(hsv, np.array([0, 50, 0]), np.array([255, 140, 60]))
    f = cv2.dilate(f, np.ones((2, 2), np.uint8), iterations=3)

    a = cv2.HoughLinesP(f, 1, pi/180, 100, 1000, 100)

    mask = np.zeros(img.shape[:2], dtype="uint8")

    for l in a:
        cv2.line(mask, (l[0][0], l[0][1]), (l[0][2], l[0][3]), 255, 10)

    railway = []

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        rect = cv2.minAreaRect( contour )
        if min(rect[1]) >= 25:
            continue
    
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        # cv2.drawContours(mask2,[box],0,(0,255,0),1)
        
        p1a = ( int((box[0][0] + box[1][0]) / 2), int((box[0][1] + box[1][1]) / 2) )
        p2a = ( int((box[2][0] + box[3][0]) / 2), int((box[2][1] + box[3][1]) / 2) )
    
        p1b = ( int((box[0][0] + box[3][0]) / 2), int((box[0][1] + box[3][1]) / 2) )
        p2b = ( int((box[1][0] + box[2][0]) / 2), int((box[1][1] + box[2][1]) / 2) )
    
        dist1 = np.linalg.norm(np.array(p1a) - np.array(p2a))
        dist2 = np.linalg.norm(np.array(p1b) - np.array(p2b))
    
        if dist1 > dist2:
            p1, p2 = p1a, p2a
        else:
            p1, p2 = p1b, p2b

        if max(dist1, dist2) < 100:
            continue
    
        railway.append([p1, p2])
    
    return railway


def draw_hud(img, height, mission_name):
    # drone name
    cv2.putText(img, 'ORT Drone', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
    # height
    cv2.putText(img, f'Висота: {height} м', (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
    # mission name
    cv2.putText(img, f'Мiсiя: {mission_name}', (10, 110), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
    # add the frame
    cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 255, 0), 5)

    return img


def run(im):
    out = im.copy()

    for p1, p2 in find_railways(im):
        print(p1, p2)
        cv2.line(out, p1, p2, (0, 255, 0), 2, 1)
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.putText(out, "залiзна дорога", (mid[0], mid[1]-20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_4)


    draw_hud(out, 500, "Спостереження")
    return out


im = cv2.imread("./samples/15.png")
out = run(im)
cv2.imwrite("out.png", out)
