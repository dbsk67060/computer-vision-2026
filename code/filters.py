import cv2  

gray = cv2.imread("../images/pills.jpg", 0)
bgr = cv2.imread("../images/zebra.jpg")
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

gauss = cv2.GaussianBlur(bgr, (11, 11), 0)
bilat = cv2.bilateralFilter(bgr, 5, sigmaColor=75, sigmaSpace=75)


def compareEdges(filteredImg):
    sobelx = cv2.Sobel(src=filteredImg, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    sobely = cv2.Sobel(src=filteredImg, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
    sobelxy = cv2.Sobel(src=filteredImg, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    canny = cv2.Canny(image=filteredImg, threshold1=100, threshold2=200)
    laplacian = cv2.convertScaleAbs(cv2.Laplacian(filteredImg, cv2.CV_64F))

    cv2.imshow("Sobel X", sobelx)
    cv2.imshow("Sobel Y", sobely)
    cv2.imshow("Sobel XY", sobelxy)
    cv2.imshow("Laplace", laplacian)
    cv2.imshow("Canny", canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def hueEdges(hsvimg):
    shift = 25;
    h, s, v = cv2.split(hsvimg)
    shiftedHue = h.copy()

    height = shiftedHue.shape[0]
    width = shiftedHue.shape[1]
    for y in range(0, height):
        for x in range(0, width):
            shiftedHue[y, x] = (h[y, x] + shift)%180

    canny = cv2.Canny(shiftedHue, 150, 255);

    cv2.imshow("Canny on shifted hue", canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def contourDetection(grayimg):
    grayimg = cv2.bilateralFilter(grayimg, 9, sigmaColor=75, sigmaSpace=75)
    ret, thresh = cv2.threshold(grayimg, 100, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    grayCopy = grayimg.copy()
    cv2.drawContours(grayCopy, contours, -1, (0,255,0), 2, cv2.LINE_AA)
    cv2.imshow("Contours", grayCopy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def showComparison():
    cv2.imshow("org", bgr)
    cv2.imshow("gauss", gauss)
    cv2.imshow("bilat", bilat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#compareEdges(bilat)
#hueEdges(hsv)
contourDetection(gray)
