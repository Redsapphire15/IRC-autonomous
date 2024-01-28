#!/usr/bin/env python
import sys
import rospy
#import rosbag
from navigation.msg import gps_data
import math
import time
import cv2
import numpy as np
import imutils
from traversal.msg import WheelRpm
from traversal.srv import *
from std_msgs.msg import Bool
import pyrealsense2 as rs
import threading
import std_msgs.msg as std_msgs
from ultralytics import YOLO
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics.utils.plotting import Annotator

class auto():

    def __init__(self):
        
        #self.cap=cv2.VideoCapture(0)
        rospy.on_shutdown(self.stop_run)
        self.pipeline = rs.pipeline()
        config = rs.config()
        rospy.init_node("arrowdetectmorethan3")
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        self.template_r=cv2.imread('Template.png',0)
        self.template_l=cv2.imread('Template_l.png',0)
        self.template_r=cv2.resize(self.template_r,(60,40),cv2.INTER_AREA)
        self.template_l=cv2.resize(self.template_l,(60,40),cv2.INTER_AREA)
        self.h,self.w=self.template_r.shape
        self.z_angle = self.x_angle = self.y_angle = 0
        self.turn = False
        self.min_dist=1.5
        self.dist_thresh=0.3
        self.angle_thresh=8
        self.kp=25
        self.kp_rot=1.5
        self.kp_straight_rot=1.5
        self.distance=0.0
        self.direction=None
        self.current_latitude=0.0
        self.current_longitude=0.0
        self.ret=False
        self.initial_yaw=0.0
        self.rotate_angle = 90
        #bag
#        self.num=i
#        filename = "imu_data_"+str(self.num)+".bag"
#        self.bag=rosbag.Bag(filename,'w')
        self.state = False
        self.first_z_angle=0

        #search alg by turning realsense
        self.enc_data=0
        self.start_time=time.time()
        self.time_thresh = 20
        self.pub = rospy.Publisher('stm_write', std_msgs.Int32MultiArray, queue_size=10)
        self.init = False
        self.start_angle = 60
        self.angle_thresh = 4
        

        try:
            rospy.Subscriber('state', Bool, self.state_callback)
            print("1")
            rospy.Subscriber('chatter',std_msgs.Float32,self.yaw_callback)
            print("2")
            rospy.Subscriber('enc_auto',std_msgs.Int8,self.enc_callback)
            print("3")
#            rospy.Subscriber('gps_coordinates', gps_data, self.gps_callback)
        except KeyboardInterrupt:
            # quit
            sys.exit()
    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        align=rs.align(rs.stream.color)
        frames=align.process(frames)
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        if not depth_frame or not color_frame:
            return False, None, None, None
        return True, depth_image, color_image, depth_frame

    def arrowdetectlessthan3(self,image,depth):
        #_,image=self.cap.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found=None
        found_l=None
        for scale in np.linspace(0.06, 1.0, 70)[::-1]:
            resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])
            if resized.shape[0] < self.h or resized.shape[1] < self.w:
                break
            #cv2.imshow("",resized)
            result = cv2.matchTemplate(resized, self.template_r, cv2.TM_CCOEFF_NORMED)
            minVal,maxVal,minLoc,maxLoc = cv2.minMaxLoc(result)
            result = cv2.matchTemplate(resized, self.template_l, cv2.TM_CCOEFF_NORMED)
            minVal_l,maxVal_l,minLoc_l,maxLoc_l = cv2.minMaxLoc(result)
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)
            if found_l is None or maxVal_l > found_l[0]:
                found_l = (maxVal_l, maxLoc_l, r)
        (maxVal, maxLoc, r) = found
        (maxVal_l, maxLoc_l, r) = found_l

        if maxVal_l>maxVal:
            maxVal=maxVal_l
            maxLoc=maxLoc_l
            direction="left"

        else:
            direction="right"

        if maxVal>0.70:
            (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
            (endX, endY) = (int((maxLoc[0]+self.w)*r), int((maxLoc[1]+self.h)*r))
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            point=(int((startX+endX)/2),int((startY+endY)/2))
            camera_info=depth.profile.as_video_stream_profile().intrinsics
            distance=depth.get_distance(point[0],point[1])
            centre=rs.rs2_deproject_pixel_to_point(camera_info,point,distance)
            distance=centre[2]
            #print(point[0])
            return True, direction,point[0],distance
        return False,None,None,None
    
    def arrowdetectmorethan3(self):
        model = YOLO('/home/kavin/Downloads/arrowspt2/runs/detect/train7/weights/best.pt')
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipeline.start(config)

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            img = np.asanyarray(color_frame.get_data())

            results = model.predict(img)

            arrow_in_center = False

            if results == None:
                arrow_detected = "Not detected"
            else:
                arrow_detected = "Detected"

            for r in results:
                annotator = Annotator(img)
                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                    c = box.cls
                    annotator.box_label(b, model.names[int(c)])

                    # Get depth data for the bounding box
                    left, top, right, bottom = map(int, b)
                    arrow_center = (left+right)/2
                    depth = depth_frame.get_distance((left + right) // 2, (top + bottom) // 2)
                    print("Depth for box" + str(b) + ":" + str(depth) +"meters")

                    # Check if the arrow is in the center along the x-axis
                    img_center_x = img.shape[1] // 2  # Calculate the center along x-axis
                    # if left <= img_center_x <= right:
                    if abs(arrow_center-img_center_x)<20:
                        arrow_in_center = True
            cv2.imshow('YOLO V8 Detection', img)
            if arrow_in_center:
                cv2.putText(img, "Arrow in centre", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                return arrow_detected, "Not Available", arrow_center, depth
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break

        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()
        return False, "Not available", None, None

#self.ret,self.direction,pix,self.distance
    def search(self):
        if(abs(self.enc_data) < self.angle_thresh and self.ret):
            return

        print("Search Called")
        print(time.time())
        print(self.start_time)
        if time.time() - self.start_time < self.time_thresh:
            print(time.time()-self.start_time)
            return
        msg1 = WheelRpm()
        msg1.hb = False
        msg1.omega = 127
        msg1.vel = 127
        wheelrpm_pub.publish(msg1)
        print("Rover Stopped")
        msg = std_msgs.Int32MultiArray()
        msg.data=[0,0,0,0,0,0]
        
        msg.layout = std_msgs.MultiArrayLayout()
        msg.layout.data_offset = 0

        msg.layout.dim = [ std_msgs.MultiArrayDimension() ]
        msg.layout.dim[0].size = msg.layout.dim[0].stride = len(msg.data)
        msg.layout.dim[0].label = 'write'
        self.pub.publish(msg)
        print("Entered While Loop")
        
        while self.init == False and abs(self.enc_data) < abs(self.start_angle)-2*self.angle_thresh :
            msg.data = [-255,0,0,0,0,0]
            rate.sleep()
            self.pub.publish(msg)
            self.start_time = time.time() - self.time_thresh
        msg.data = [0,0,0,0,0,0]
        print ("Exited While Loop")
        self.init = True
        print(self.init, self.enc_data, self.ret)
        if self.init == True and abs(self.enc_data) < abs(self.start_angle)-self.angle_thresh and not self.ret:
            print("Camera Moving")
            msg.data = [255,0,0,0,0,0]
            rate.sleep()
            self.pub.publish(msg)
        elif self.init == True and abs(self.enc_data) < abs(self.start_angle) and self.ret:
            msg.data = [0,0,0,0,0,0]
            rate.sleep()
            print("Arrow found at: ", self.enc_data)
            if(self.enc_data <0):
                self.direction = 'right'
                self.rotate_angle = abs(self.enc_data+self.angle_thresh)
                self.turn = True
                self.initial_yaw=self.z_angle
            else:
                self.direction = 'left'
                self.rotate_angle = abs(self.enc_data+self.angle_thresh)
                self.turn = True
                self.initial_yaw=self.z_angle 
            self.start_time = time.time()
            self.init = False
            while abs(self.enc_data) > self.angle_thresh :
                if(self.enc_data < 0):
                    msg.data = [255,0,0,0,0,0]
                else:
                    msg.data = [-255,0,0,0,0,0]
                rate.sleep()
                self.pub.publish(msg)
            msg.data = [0,0,0,0,0,0]
            self.pub.publish(msg)
            self.init = False
            self.start_time = time.time()

        elif not self.ret:
            while abs(self.enc_data) > self.angle_thresh :
                if(self.enc_data < 0):
                    msg.data = [255,0,0,0,0,0]
                else:
                    msg.data = [-255,0,0,0,0,0]
                rate.sleep()
                self.pub.publish(msg)
            msg.data = [0,0,0,0,0,0]
            self.pub.publish(msg)
            self.init = False
            self.start_time = time.time()
        '''     
            while self.enc_data >5:
                pass
                #go in one direction to 0.
            while self.enc_data < -5:
                pass
                #go in other direction to 0.
            return
        '''

    def main(self):
        
#        self.write_to_bagfile()
        if(not self.turn):
            self.search()
        if(not self.turn):
            ret, depth_frame, color_frame, depth = self.get_frame()
            self.ret,self.direction,pix,self.distance=self.arrowdetectmorethan3()
            if self.distance<3.14:
                self.ret,self.direction,pix,self.distance=self.arrowdetectlessthan3(color_frame,depth)
                if self.ret==False:
                    self.ret,self.direction,pix,self.distance=self.arrowdetectmorethan3()
            if (self.ret):
                print("arrow detected at distance: "+str(self.distance))
                print("Direction: " + self.direction)
            else:
                print("Trying to detect arrow...")

            self.move_straight()
        else:
            if(self.direction=="left"):
                self.rotate(1)
            else:
                self.rotate(-1)

    def spin(self):
        while not rospy.is_shutdown():
            if(self.state==True):
                self.main()
                rate.sleep()
            else:
                print("Rover in Joystick mode")
                rate.sleep()
  

    def state_callback(self,msg):
        self.state=msg.data

    def yaw_callback(self,msg):
        if(not self.first_z_angle):
            self.first_z_angle=self.z_angle
        self.z_angle = msg.data

    def enc_callback(self,msg):
        self.enc_data = msg.data

#    def write_to_bagfile(self):
#        imu_dat=imu_angle()
#        imu_dat.Yaw=self.z_angle
#        imu_dat.Roll=self.distance
#        imu_dat.Pitch=self.distance
#        self.bag.write('/imu_angles',imu_dat)
#
#        gps_dat=gps_data()
#        gps_dat.latitude=self.current_latitude
#        gps_dat.longitude=self.current_longitude
#        self.bag.write('/gps_coordinates',gps_dat)

#    def gps_callback(self,msg):
#        if(msg.latitude and  msg.longitude):
#            self.current_latitude=msg.latitude
#            self.current_longitude=msg.longitude

#    def write_coordinates(self):
#        file_object=open("coordinates.txt","a")
#        file_object.write("latitude: "+str(self.current_latitude)+" longitude: "+ str(self.current_longitude))
#        file_object.close()

    def move_straight(self):
        msg =WheelRpm()
        msg.hb=False
        msg.omega=127
        # if(self.first_z_angle):
           # msg.omega=127-self.kp_straight_rot*(self.first_z_angle-self.z_angle)
        # else:
           # msg.omega =127
        #print(self.init)
        if(self.init):
            #print("if condition entered")
            msg.vel=127
            wheelrpm_pub.publish(msg)
        elif(self.ret):
            if(abs(self.min_dist-self.distance)>self.dist_thresh):
                msg.vel=max(102,127+self.kp*(self.min_dist-self.distance))
                wheelrpm_pub.publish(msg)
            else:
                msg.vel=127
                wheelrpm_pub.publish(msg)
                print("Stopped going Straight")
                rospy.sleep(10)   # 10s   # Competition rules say 10s
                self.turn = True
                self.rotate_angle = 90
#                self.write_coordinates()
                self.initial_yaw=self.z_angle

        else:
            print("Forward (WRONG)")
            msg.vel = 95
            wheelrpm_pub.publish(msg)
            self.turn=False

    def rotate(self,dir):
        msg = WheelRpm()
        msg.vel = 127
        msg.hb=False
        diff  = self.z_angle - self.initial_yaw
        if (diff > 120):
            diff = diff - 360
        elif (diff < -120):
            diff = diff + 360
        print("diff=",diff)
        '''
        if (-60<self.z_angle-self.initial_yaw<60):
            error=30
        else:
            error = 90-abs(self.z_angle-self.initial_yaw)
        '''
        error = self.rotate_angle-abs(diff)
        print("error=", error)
        if (abs(error)>self.angle_thresh):
            msg.omega=127+(dir*40)
            msg.vel=102
            print("Calling Rotate, printing Z angle below")
            print(error)
            wheelrpm_pub.publish(msg)
        else:
            msg.omega=127
            msg.vel = 92
            wheelrpm_pub.publish(msg)
            self.z_angle=0
            print("****ROTATE DONE*****")
            self.turn=False
            rospy.sleep(2)

    def stop_run(self):
        msg_stop=WheelRpm()
        msg_stop.hb=False
        msg_stop.vel=msg_stop.omega=127
        wheelrpm_pub.publish(msg_stop)
#        self.bag.close()

if __name__ == '__main__':
    try:
        rospy.init_node("arrowdetect")
        rate=rospy.Rate(10)
#    i=int(input("Enter test number: "))
        wheelrpm_pub=rospy.Publisher('motion',WheelRpm,queue_size=10)
        run=auto()
        run.spin()
    except KeyboardInterrupt:
    # quit
        sys.exit()
