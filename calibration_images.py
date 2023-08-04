import cv2


cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

num = 0


while cap.isOpened():

    # Left Cam
    succes1, img = cap.read()

    # Right Cam
    succes2, img2 = cap2.read()

    k = cv2.waitKey(1)

    if k == 27:
        break
    elif k == ord('s'):  # wait for 's' key to save and exit
        cv2.imwrite('calibration_images/stereoLeft/imageL' +
                    str(num) + '.png', img)
        cv2.imwrite('calibration_images/stereoRight/imageR' +
                    str(num) + '.png', img2)
        print("images saved!")
        num += 1
    cv2.imshow('Left Cam', img)
    cv2.imshow('Right Cam', img2)
