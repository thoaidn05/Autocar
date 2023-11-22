import socket       
import sys      
import time
import cv2
import numpy as np
import json
import base64
from detectLane import *
import math

# Create a socket object 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 

# Define the port on which you want to connect 
port = 54321                
  
# connect to the server on local computer 
s.connect(('127.0.0.1', port)) 
angle = 10
speed = 100

error_arr = np.zeros(5)
pre_t = time.time()
MAX_SPEED = 100

def get_angle(img, left, right):
    Oy , AD = img.shape[:2]
    AD = AD/2
    OA = left[1]
    DE = Oy - OA
    AO = (left[0]+right[0])/2
    DO = AO - AD
    angle = math.atan(DO/DE) * 180 / math.pi
    return int(angle)



def PID(error, p, i, d):
    global pre_t
    # global error_arr
    error_arr[1:] = error_arr[0:-1]
    error_arr[0] = error
    P = error*p
    delta_t = time.time() - pre_t
    #print('DELAY: {:.6f}s'.format(delta_t))
    pre_t = time.time()
    D = (error-error_arr[1])/delta_t*d
    I = np.sum(error_arr)*delta_t*i
    angle = P + I + D
    if abs(angle)>25:
        angle = np.sign(angle)*25
    return int(angle)

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
            current_angle = data_recv["Angle"]
            current_speed = data_recv["Speed"]
            # print("angle: ", current_angle)
            # print("speed: ", current_speed)
            # print("---------------------------------------")
            #Img data recv from server
            jpg_original = base64.b64decode(data_recv["Img"])
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            _image = cv2.imdecode(jpg_as_np, flags=1)
            image,left_lane, right_lane = frame_processor(_image)
            _angle = get_angle(image,left_lane, right_lane)
            angle = PID(_angle, p=1.0, i=0.1, d=0.01)
            print(angle)
            cv2.imshow("IMG", image)
            # print("Img Shape: ",image.shape)
            cv2.waitKey(1)
    finally:
        print('closing socket')
        s.close()

