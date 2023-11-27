import socket       
import sys      
import time
import cv2
import numpy as np
import json
import base64
import math

# Create a socket object 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 

# Define the port on which you want to connect 
port = 54321                
  
# connect to the server on local computer 
s.connect(('127.0.0.1', port)) 

count = 16
angle = 0
speed = 150

timestart = 0
last_angle = angle

def region_selection(image):
	"""
	Determine and cut the region of interest in the input image.
	Parameters:
		image: we pass here the output from canny where we have 
		identified edges in the frame
	"""
	# create an array of the same size as of the input image 
	mask = np.zeros_like(image) 
	# if you pass an image with more then one channel
	if len(image.shape) > 2:
		channel_count = image.shape[2]
		ignore_mask_color = (255,) * channel_count
	# our image only has one channel so it will go under "else"
	else:
		# color of the mask polygon (white)
		ignore_mask_color = 255
	# creating a polygon to focus only on the road in the picture
	# we have created this polygon in accordance to how the camera was placed
	rows, cols = image.shape[:2]
	bottom_left = [cols * 0, rows * 1]
	top_left	 = [cols * 0, rows * 0.25]
	bottom_right = [cols * 1, rows * 1]
	top_right = [cols * 1, rows * 0.25]
	vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
	# filling the polygon with white color and generating the final mask
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	# performing Bitwise AND on the input image and mask to get only the edges on the road
	masked_image = cv2.bitwise_and(image, mask)
	return masked_image

if __name__ == "__main__":
    try:
        """
            - Chương trình đưa cho bạn 3 giá trị đầu vào:
                * image: hình ảnh trả về từ xe
                * current_speed: vận tốc hiện tại của xe
                * current_angle: góc bẻ lái hiện tại của xe
            - Bạn phải dựa vào giá trị đầu vào này để tính toán và
            gán lại góc lái và tốc độ xe vào 2 biến:
                * Biến điều khiển: sendBack_angle, sendBack_Speed
                Trong đó:
                    + sendBack_angle (góc điều khiển): [-25, 25]
                        NOTE: ( âm là góc trái, dương là góc phải)
                    + sendBack_Speed (tốc độ điều khiển): [0, 150]
            """
        while True:
            # Gửi góc lái và tốc độ để điều khiển xe
            message = bytes(f"{angle} {speed}", "utf-8")
            s.sendall(message)

            # Recive data from server
            data = s.recv(100000)
            # print(data)
            data_recv = json.loads(data)

            # Angle and speed recv from server
            # current_angle = data_recv["Angle"]
            # current_speed = data_recv["Speed"]
            # print("angle: ", current_angle)
            # print("speed: ", current_speed)
            # print("---------------------------------------")
            #Img data recv from server
            jpg_original = base64.b64decode(data_recv["Img"])
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            imgage = cv2.imdecode(jpg_as_np, flags=1)
            
            ig = imgage
            
            imgage = imgage[200:,:,:]

            lower_green = np.array([95,0,0], dtype = "uint8")
            upper_green = np.array([255,255,255], dtype = "uint8")
            mask = cv2.inRange(imgage, lower_green, upper_green)
            lane = cv2.bitwise_and(imgage, imgage, mask = mask)

            gray = cv2.cvtColor(lane, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7,7), 0)
            canny = cv2.Canny(blur,95,255)

            img = region_selection(canny)

            arr = []
            linerow = img[50,:]
            for x,y in enumerate(linerow):
                if y == 255:
                    arr.append(x)
            arrmax = max(arr)
            arrmin = min(arr)
            center = int((arrmax+arrmin)/2)
            angle = math.degrees(math.atan((center-img.shape[1]/2)/(img.shape[0]-50)))
            
            
            if (angle > 10): angle = 10
            elif (angle < -10): angle = -10

            if (np.abs(angle - last_angle) >= 5) :
                 speed = 50
                 last_angle = angle
            else: 
                 speed = 150

            print(angle)

            cv2.circle(img,(arrmin,50),5,(255,255,255),5)
            cv2.circle(img,(arrmax,50),5,(255,255,255),5)
            cv2.line(img,(center,50),(int(img.shape[1]/2),img.shape[0]),(255,255,255),(5))
            cv2.imshow("IMG", img)
            # print("Img Shape: ",imgage.shape)
            #save image
            # if time.time()-timestart > 1:
            #     image_name = "./img/img_{}.jpg".format(count)
            #     count += 1
            #     timestart = time.time()
            #     cv2.imwrite(image_name, ig)
            key = cv2.waitKey(1)
        
            

    finally:
        print('closing socket')
        s.close()

