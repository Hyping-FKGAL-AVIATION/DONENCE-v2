from dronekit import connect
from time import sleep
from keyboard import is_pressed
from pymavlink import mavutil
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

sleep_duration = 1/10
needed_height = 20
photonumber = 1
last_loc = None
cur_loc = None
Between_2_loc = 5
cap = cv2.VideoCapture(0)
os.chdir(r"C:\Users\Pc\Desktop\Yolo Kodları\Otonom görev\Test\Take_pics\Pictures")

def send_green_dot_signal():
    """
    Sends a 'Camera Trigger' signal to the Flight Controller.
    On SITL: Draws a green dot on the map.
    On Real Drone: Logs the CAM event in the DataFlash logs for geotagging.
    """
    msg = vehicle.message_factory.command_long_encode(
        0, 0,    # target_system, target_component
        mavutil.mavlink.MAV_CMD_DO_DIGICAM_CONTROL, # Command
        0,       # confirmation
        0,       # param1 (Session control)
        0,       # param2 (Zoom Absolute)
        0,       # param3 (Zoom Relative)
        0,       # param4 (Focus)
        1,       # param5 (Shoot Command: 1 = Shoot)
        0,       # param6 (Command Identity)
        0        # param7 (Shot ID)
    )
    vehicle.send_mavlink(msg)
    print("Green Dot")

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
            print(f"Distance = {Distance}") #, last_loc = {last_loc}, cur_loc = {cur_loc}")
            if Distance >= Between_2_loc:
                print("Shot a picture! Resetting distance values...")
                takepicture()
                send_green_dot_signal()
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