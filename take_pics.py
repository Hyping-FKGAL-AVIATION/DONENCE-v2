from dronekit import connect
from time import sleep
from keyboard import is_pressed
import math
import os
import cv2

connection_string = "tcp:127.0.0.1:5762"

print(f"Connecting to vehicle on: {connection_string}")

try:
    vehicle = connect(connection_string, wait_ready=True, timeout=60)
except Exception as e:
    print(f"Failed to connect: {e}")
    exit() #exit() fonksiyonu tüm python kodunu durdurur.

print("Connected to the vehicle")

sleep_duration = 1/100
needed_height = 20
photonumber = 1
last_loc = None
cur_loc = None
Between_2_loc = 5
cap = cv2.VideoCapture(0)
os.chdir(r"C:\Users\Pc\Desktop\Yolo Kodları\Otonom görev\Test\Take_pics\Pictures")

def takepicture():
    boolean, frame = cap.read()
    if boolean:
        cv2.imwrite(f"Webcam{photonumber}.jpg", frame)
    
while True:
    if vehicle.armed == True:
        location = vehicle.location.local_frame
        height = -location.down
        if height < needed_height + 1.5 and height > needed_height - 1.5:
            east = location.east
            north = location.north
            if last_loc == None:
                last_loc = [east, north]
            cur_loc = [east, north]
            last_east = last_loc[0]
            last_north = last_loc[1]
            cur_east = cur_loc[0]
            cur_north = cur_loc[1]
            Distance = math.sqrt((cur_north - last_north) ** 2 + (cur_east - last_east) ** 2)
            print(f"Distance = {Distance}, last_loc = {last_loc}, cur_loc = {cur_loc}")
            if Distance >= Between_2_loc:
                print("Shot a picture! Resetting distance values...")
                takepicture()
                last_loc = cur_loc
                photonumber += 1
        else:
            print(f"Vehicle is on but it is not in the right height level. Height: {height} Maximum Height:{needed_height + .5} Minimum Height: {needed_height - .5}")
    else:
        cur_loc = None
        last_loc = None
        print("Vehicle is not armed")
    if is_pressed("x"):
        break
    sleep(sleep_duration)

print("loop ended")