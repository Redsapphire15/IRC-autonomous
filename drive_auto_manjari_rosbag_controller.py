#!/usr/bin/env python

import rospy
import rosbag
from navigation.msg import gps_data
import math
from time import *
import cv2
import numpy as np
import imutils
from navigation.msg import imu_angle
from traversal.msg import WheelRpm
from traversal.srv import *
from std_msgs.msg import Bool
import pyrealsense2 as rs

class auto():

    def __init__(self,i):
        
        #self.cap=cv2.VideoCapture(0)
        rospy.on_shutdown(self.stop_run)
        self.pipeline = rs.pipeline()
        config = rs.config()
        rospy.init_node("arrowdetect")
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
        #self.rotate_incr = 0
        self.turn = False
        self.min_dist=1.7
        self.dist_thresh=0.3
        self.angle_thresh=3
        self.kp=25
        self.kp_rot=1.0
        self.kp_straight_rot=1.5
        self.distance=0.0
        self.direction=None
        self.current_latitude=0.0
        self.current_longitude=0.0
        self.ret=False
        self.initial_yaw=0.0
        self.num=i
        filename = "imu_data_"+str(self.num)+".bag"
        self.bag=rosbag.Bag(filename,'w')
        self.state = False
        self.first_z_angle=0


        try:
            rospy.Subscriber('state', Bool, self.state_callback)
            rospy.Subscriber('imu_angles', imu_angle, self.angles_callback)
            rospy.Subscriber('gps_coordinates', gps_data, self.gps_callback)
        except Exception(e):
            print("error")

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

    def arrowdetect(self,image,depth):
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
            direction='left'

        else:
            direction='right'

        if maxVal>0.35:
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
        return False,None,None,0.0

    def main(self):
        self.write_to_bagfile()
        if(not self.turn):
            ret, depth_frame, color_frame, depth = self.get_frame()
            self.ret,self.direction,pix,self.distance=self.arrowdetect(color_frame,depth)
            if (self.ret):
                print("arrow detected at distance: "+str(self.distance))
                print(self.direction)
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

    def angles_callback(self,msg):
        if(not self.first_z_angle):
            self.first_z_angle=self.z_angle
        self.z_angle = msg.Yaw
        self.x_angle=msg.Roll
        self.y_angle=msg.Pitch

    def write_to_bagfile(self):
        imu_dat=imu_angle()
        imu_dat.Yaw=self.z_angle
        imu_dat.Roll=self.distance
        imu_dat.Pitch=self.distance
        self.bag.write('/imu_angles',imu_dat)

        gps_dat=gps_data()
        gps_dat.latitude=self.current_latitude
        gps_dat.longitude=self.current_longitude
        self.bag.write('/gps_coordinates',gps_dat)

    def gps_callback(self,msg):
        if(msg.latitude and  msg.longitude):
            self.current_latitude=msg.latitude
            self.current_longitude=msg.longitude

    def write_coordinates(self):
        file_object=open("coordinates.txt","a")
        file_object.write("latitude: "+str(self.current_latitude)+" longitude: "+ str(self.current_longitude))
        file_object.close()

    def move_straight(self):
        msg =WheelRpm()
        msg.hb=False
        msg.omega=127
        # if(self.first_z_angle):
           # msg.omega=127-self.kp_straight_rot*(self.first_z_angle-self.z_angle)
        # else:
           # msg.omega =127 
        if(self.ret):
            if(abs(self.min_dist-self.distance)>self.dist_thresh):
                msg.vel=127+self.kp*(self.min_dist-self.distance)
                wheelrpm_pub.publish(msg)
            else:
                msg.vel=127
                wheelrpm_pub.publish(msg)
                print("Stopped going Straight")
                rospy.sleep(10)   # 10s   # Competition rules say 10s
                self.turn = True
                self.write_coordinates()
                self.initial_yaw=self.z_angle

        else:
            msg.vel = 95
            wheelrpm_pub.publish(msg)
            self.turn=False

    def rotate(self,dir):
        msg = WheelRpm()
        msg.vel = 127
        msg.hb=False
        error = 90-abs(self.z_angle-self.initial_yaw)

        if(abs(error)>self.angle_thresh):
            msg.omega=127+(dir*(8*np.sign(error)+self.kp_rot*error))
            print("Calling Rotate, printing Z angle below")
            print(error)
            wheelrpm_pub.publish(msg)
        else:
            msg.omega=127
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
        self.bag.close()

if __name__ == '__main__':
    rospy.init_node("arrowdetect")
    rate=rospy.Rate(10)
    i=int(input("Enter test number: "))
    wheelrpm_pub=rospy.Publisher('motion',WheelRpm,queue_size=10)
    run=auto(i)
    run.spin()
