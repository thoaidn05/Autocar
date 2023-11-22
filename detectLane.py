import numpy as np
import cv2
import matplotlib.pyplot as plt

def find_mid_lane(contours):
    xlane = []
    ylane = 55
    if len(contours) == 2:
        for cnt in contours:
            for point in cnt:
                x, y = point[0]
                if y == ylane:
                    xlane.append(x)
    elif len(contours) < 2:
        left_lane = left_lane
        right_lane = right_lane
        
    left_lane = (xlane[0], ylane)
    right_lane = (xlane[-1], ylane)
    return left_lane, right_lane

def frame_processor(image):
    image = image[200:,:]
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    lower_tree = np.array([190,0,0], dtype= "uint8")
    upper_tree = np.array([255,255,255], dtype= "uint8")
    tree_mask = cv2.inRange(sobel_y, lower_tree, upper_tree)

    kernel = np.ones((9,9), np.uint8)
    dilation = cv2.dilate(tree_mask,kernel=kernel,iterations = 6)

    kernel = np.ones((9,9),np.uint8)
    erosion = cv2.erode(dilation,kernel,iterations = 3)

    road = cv2.bitwise_and(blurred, blurred, mask=erosion)
    lower = np.array([115,100,90], dtype= "uint8")
    upper = np.array([255, 115, 255], dtype= "uint8")
    shadow_lane = cv2.inRange(road,lower,upper)

    road_gray = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
    thresh = cv2.inRange(road_gray,128,210)
    mask = cv2.bitwise_or(shadow_lane,thresh)

    kernel = np.ones((5,5), np.uint8)
    dilation = cv2.dilate(mask,kernel=kernel,iterations = 2)

    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(dilation,kernel,iterations = 2)

    lane = cv2.bitwise_and(road_gray, road_gray, mask=erosion)

    contours, _ = cv2.findContours(lane, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 600]
    left_lane, right_lane =find_mid_lane(filtered_contours)
    cv2.drawContours(image, filtered_contours, -1, (0, 255, 0), 3)

    return image, left_lane, right_lane

# image_name = "./img/test.jpg"
# _image = cv2.imread('./img/_img_18.jpg')
# image, left_lane, right_lane = frame_processor(_image)
# cv2.imwrite(image_name, image)

# plt.imshow(image)
# plt.scatter(left_lane[0], left_lane[1], color="red")
# plt.scatter(right_lane[0], right_lane[1], color="red")
# plt.show()