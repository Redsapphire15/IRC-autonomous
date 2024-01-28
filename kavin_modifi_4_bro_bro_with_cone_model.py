#!/usr/bin/env python3
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
from collections import defaultdict

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
        self.circle_dist=1.5
        self.dist_thresh=0.3
        self.angle_thresh=4
        self.kp=20
        self.kp_rot=1.5
        self.kp_straight_rot=7.5
        self.distance=10.0
        for i in range(5):
            print("hey! self.distance = 10",self.distance)
        self.direction="Not Available"
        self.current_latitude=0.0
        self.current_longitude=0.0
        self.ret=False
        self.initial_yaw=0.0
        self.rotate_angle = 90
        self.angles_dict = defaultdict(list)
        self.searchcalled = False
        self.latlong = defaultdict(list)
        self.latlong[0] = "latitude"
        self.latlong[1] = "longitude"
        self.arrow_numbers = 0
        self.gpscalled =0
        

        #bag
#        self.num=i
#        filename = "imu_data_"+str(self.num)+".bag"
#        self.bag=rosbag.Bag(filename,'w')
        self.state = False
        self.initial_drift_angle=0

        #search alg by turning realsense
        self.enc_data=0
        self.start_time=time.time()
        self.time_thresh = 20
        self.pub = rospy.Publisher('stm_write', std_msgs.Int32MultiArray, queue_size=10)
        self.init = False
        self.start_angle = 55
        self.angle_thresh = 4
        #self.manjari = False
        self.count_arrow = 0
        try:
            rospy.Subscriber('state', Bool, self.state_callback)
            print("1")
            rospy.Subscriber('chatter',std_msgs.Float32,self.yaw_callback)
            print("2")
            rospy.Subscriber('enc_auto',std_msgs.Int8,self.enc_callback)
            print("3")  
            rospy.Subscriber('gps_coordinates', gps_data, self.gps_callback)
            print("4")
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
        # self.manjari = True
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
            try:
                distance=depth.get_distance(point[0],point[1])
            except:
                print(point)
                self.ret = False
                #self.search()
                distance = 2.5
            centre=rs.rs2_deproject_pixel_to_point(camera_info,point,distance)
            distance=centre[2]
            #print(point[0])
            return True, direction,point[0],distance
        # self.start_time = time.time()
        return False,None,None,2.5
    


    def cone(self,image):
        hsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,(0,150,20),(15,255,255))
        kernel=np.ones((5,5))
        img_thresh_opened = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
        img_thresh_blurred = cv2.medianBlur(img_thresh_opened, 5)
        thresh=img_thresh_blurred
        contours, _= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull=[]
        bounding_rect=[]
        for c in contours:
            approx=cv2.approxPolyDP(c,0.025*cv2.arcLength(c,True),True)
            hull.append(cv2.convexHull(approx))
            bounding_rect=[]
            for ch in hull:
                if self.convex_hull_pointing_up(ch):
                    bounding_rect.append(cv2.boundingRect(ch))
                    x, y, w, h = cv2.boundingRect(ch)
                    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        return bounding_rect
        
    def arrowdetectmorethan3(self):
        model = YOLO('/home/kavin/Downloads/arrowspt2/runs/detect/train7/weights/best.pt') #old model
        # model = YOLO('/home/kavin/Downloads/best.pt')
        #model = YOLO('/home/kavin/Downloads/arrowspt2/runs/detect/train8/weights/best.pt') #new model
        '''
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipeline.start(config)
        '''
        while True:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            img = np.asanyarray(color_frame.get_data())

            results = model.predict(img, conf = 0.5, max_det = 2)
            depth = 0
            arrow_in_center = False
            arrow_center = None
            if results == None:
                arrow_detected = "Not detected"
            else:
                arrow_detected = "Detected"
            
            #arrow_in_center = False

            for r in results:
                annotator = Annotator(img)
                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                    c = box.cls
                    annotator.box_label(b, model.names[int(c)])

                    # Get dep/home/kavin/caesar2020_nvidia/src/navigation/scripts/kavin_modifi_2.pyth data for the bounding box
                    left, top, right, bottom = map(int, b)
                    arrow_center = (left+right)/2
                    try:
                        depth = depth_frame.get_distance((left + right) // 2, (top + bottom) // 2)
                    except:
                        self.ret = False
                        self.search()
                    #print("Depth for box" + str(b) + ":" + str(depth) +"meters")

                    # Check if the arrow is in the center along the x-axis
                    img_center_x = 320
                    # img_center_x = img.shape[1] // 2
                    # Calculate the center along x-axis
                    #if left <= img_center_x <= right:
                    #print("arrow coordinates relative to center ", arrow_center-img_center_x)
                    if ((arrow_center-img_center_x)>-150) and (arrow_center - img_center_x)<125 or depth<3 :
                    # if depth<25:
                        arrow_in_center = True
                    else:
                        arrow_in_center = False
                    print(arrow_center-img_center_x)
            # print("imshow")
            cv2.imshow('YOLO V8 Detection', img)   
            if cv2.waitKey(1) & 0xFF == ord(' '):
               break         
            
            print("arrow in center: ", arrow_in_center)  
            if arrow_in_center:
                #cv2.putText(img, "Arrow in centre", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print("Depth from more than 3:", depth)
                return arrow_in_center, "Not Available", arrow_center, depth
            else:
                return False, "Not available", None, 2.5
            
            
            
            print("imshow")
            cv2.imshow('YOLO V8 Detection', img)            
            #return arrow_detected, "Not Available", arrow_center, depth
        # Stop streaming
        self.pipeline.stop()
        cv2.destroyAllWindows()
        #return False, "Not available", None, 0.0

#self.ret,self.direction,pix,self.distance
    def search(self):
        print("self.searchcalled (should print false here always):",self.searchcalled)
        if(abs(self.enc_data) < 0.6*self.angle_thresh and self.ret and not self.init):   
            # if arrow is detected and realsense is facing straight, then come out of search immediately.
            return

        print("Search() has been called.")
        #self.searchcalled = True
        print("time.time():",time.time())
        print("self.start_time:",self.start_time)
        if time.time() - self.start_time < self.time_thresh:    #time_thresh is 20s
            print("time.time()-self.start_time (when this becomes 20s, search will happen):",time.time()-self.start_time)
            return
        msg1 = WheelRpm()
        msg1.hb = False
        msg1.omega = 127
        msg1.vel = 127
        wheelrpm_pub.publish(msg1)
        print("Rover has stopped.")
        msg = std_msgs.Int32MultiArray()
        msg.data=[0,0,0,0,0,0]
        
        msg.layout = std_msgs.MultiArrayLayout()
        msg.layout.data_offset = 0

        msg.layout.dim = [ std_msgs.MultiArrayDimension() ]
        msg.layout.dim[0].size = msg.layout.dim[0].stride = len(msg.data)
        msg.layout.dim[0].label = 'write'
        self.pub.publish(msg)

        print("Entered while loop.")
        while self.init == False and abs(self.enc_data) < abs(self.start_angle)-2*self.angle_thresh:   
            #to make the realsense go to the 60 degree maximum before starting the burst search

            msg.data = [0,255,0,0,0,0]
            rate.sleep()
            self.pub.publish(msg)
            self.start_time = time.time() - self.time_thresh
        msg.data = [0,0,0,0,0,0]
        print ("Exited while loop.")
        self.init = True
        print("self.init (set to true in the previous line:",self.init)
        print("Realsense's angle:", self.enc_data)
        print("self.ret:", self.ret)
        if self.init == True and abs(self.enc_data) < (abs(self.start_angle)-0.5*self.angle_thresh) and not self.ret:
            # if arrow is not detected and the realsense has not gone beyond the 60 degree maximum, continue moving realsense
            # self.init is not made false, so this will happen again when search is called in main()

            print("Camera Moving")
            msg.data = [0,-255,0,0,0,0]
            rate.sleep()
            self.pub.publish(msg)
            print()
            #main area
        elif self.init == True and abs(self.enc_data) < abs(self.start_angle) and self.ret:
            #if arrow is detected and the realsense is within the 60 degree maximum, append the arrow's values and continuemoving realsense
            # self.init is not made false, so this will happen again when search is called in main()

            #self.distance = self.distance*1
            #self.distance = self.distance // 1
            #self.distance = self.distance / 1
            self.distance = float(round(self.distance))
            if self.distance < 0.0:                                              #change
                self.angles_dict[self.distance].append(self.enc_data) 
                print("I'm appending to angles_dict")
            msg.data = [0,-255,0,0,0,0]
            self.pub.publish(msg)
            rate.sleep()
            print("Arrow found at: ", self.enc_data)
            print()
        elif not self.ret:
            # when the realsense has crossed the 60 degree maximum, realsense comes back to middle and the dictionary is processed
            # self.init is set to false (when next search() is called, realsense will first move to the 60 degree maximum)
            # and the counting of time is reset (that is, the next search will happen at least 20s after this block of code)

            while abs(self.enc_data) > self.angle_thresh :
                if(self.enc_data > 0):
                    msg.data = [0,-255,0,0,0,0]
                else:
                    msg.data = [0,255,0,0,0,0]
                rate.sleep()
                self.pub.publish(msg)
            msg.data = [0,0,0,0,0,0]
            self.pub.publish(msg)
            self.init = False
            self.searchcalled = True
            self.distance = 10.0
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
    def process_dict(self):
        print("self.searchcalled (should print true always):", self.searchcalled)
        if not self.init and self.searchcalled:
            # the first part is not needed, because whenever self.searchcalled is set to true, self.init is set to false
            # actually, this if is only not needed, because in main(), there is an if self.searchcalled(), then process_dict(), which takes care of everything

            print("the dictionary with dist:[enc angles] :- ",self.angles_dict)
            self.searchcalled = False
            max_length_key = 0.0
        
            try:
                print("Entered Try of proces_dict")
                #if min(self.angles_dict.keys()) == '0.0':
                    #del self.angles_dict['0.0']
                if min(self.angles_dict.keys()) != 0:
                    print("Entered if")
                    #del self.angles_dict['0.0']
                    max_length_key = min(self.angles_dict.keys())
                    #max_length_key = max(self.angles_dict, key=lambda k: len(self.angles_dict[k]))
                    self.min_dist=max_length_key
            except:
                print("The list is empty. No minimum value.")
                self.init = False
                self.searchcalled = False
                return
            if len(self.angles_dict[max_length_key+1])!=0:
                self.which_enc_angle_to_turn=(sum(self.angles_dict[self.min_dist])+sum(self.angles_dict[self.min_dist+1]))/(len(self.angles_dict[self.min_dist]) + len(self.angles_dict[self.min_dist + 1])) 
            else: 
                self.which_enc_angle_to_turn=sum(self.angles_dict[self.min_dist])/len(self.angles_dict[self.min_dist])
            print("Angle to turn:", self.which_enc_angle_to_turn)
            print("the dictionary with dist:[enc angles] :- ",self.angles_dict)

            #encoder need not be perfect. if in case there is some cup, edit this angle as per your needs
            if (self.which_enc_angle_to_turn<0):
                self.direction="left"
                #self.rotate_angle=abs(self.which_enc_angle_to_turn + 0.5*self.angle_thresh) #+2 degrees
                self.rotate_angle=abs(self.which_enc_angle_to_turn)
            else:
                self.direction="right"
                #self.rotate_angle=(self.which_enc_angle_to_turn - 0.5*self.angle_thresh) #+2 degrees
                self.rotate_angle=abs(self.which_enc_angle_to_turn)
        
            self.turn = True
            self.initial_yaw=self.z_angle 
            self.angles_dict = defaultdict(list)

            #self.init = False

    def convex_hull_pointing_up(self,ch):
        points_above_center, points_below_center = [], []
        x, y, w, h = cv2.boundingRect(ch)
        aspect_ratio = w / h
        if aspect_ratio < 0.7 and (w*h)>10000:
            vertical_center = y + h / 2
            for point in ch:
                if point[0][1] < vertical_center:
                    points_above_center.append(point)
                elif point[0][1] >= vertical_center:
                    points_below_center.append(point)
                if len(points_below_center)>0 and len(points_above_center)>0:
                      left_x = points_below_center[0][0][0]
                      right_x = points_below_center[0][0][0]
                else:
                      return False
                for point in points_below_center:
                    if point[0][0] < left_x:
                        left_x = point[0][0]
                    if point[0][0] > right_x:
                        right_x = point[0][0]
                for point in points_above_center:
                    if (point[0][0] < left_x) or (point[0][0] > right_x):
                        return False
                else:
                    return False
        return True


    def cone_model(self):
        model = YOLO('/home/kavin/Downloads/Safety_Cone_detection-main/Models/Cone.pt') #old model
        # model = YOLO('/home/kavin/Downloads/best.pt')
        #model = YOLO('/home/kavin/Downloads/arrowspt2/runs/detect/train8/weights/best.pt') #new model
        '''
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipeline.start(config)
        '''
        print("Cone Model Entered")
        while True:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            img = np.asanyarray(color_frame.get_data())

            results = model.predict(img, conf = 0.5, max_det = 2)
            depth = 0
            cone_in_center = False
            cone_center = None
            if results == None:
                print("Cone Not detected")
            else:
                print("Detected")
            
            #arrow_in_center = False

            for r in results:
                annotator = Annotator(img)
                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                    c = box.cls
                    annotator.box_label(b, model.names[int(c)])

                    # Get dep/home/kavin/caesar2020_nvidia/src/navigation/scripts/kavin_modifi_2.pyth data for the bounding box
                    left, top, right, bottom = map(int, b)
                    cone_center = (left+right)/2
                    try:
                        depth = depth_frame.get_distance((left + right) // 2, (top + bottom) // 2)
                    except:
                        self.ret = False
                        self.search()
                    #print("Depth for box" + str(b) + ":" + str(depth) +"meters")

                    # Check if the arrow is in the center along the x-axis
                    img_center_x = 320
                    # img_center_x = img.shape[1] // 2
                    # Calculate the center along x-axis
                    #if left <= img_center_x <= right:
                    #print("arrow coordinates relative to center ", arrow_center-img_center_x)
                    if ((cone_center-img_center_x)>-150) and (cone_center - img_center_x)<125 or depth<3 :
                    # if depth<25:
                        cone_in_center = True
                    else:
                        cone_in_center = False
                    print(cone_center-img_center_x)
            # print("imshow")
            cv2.imshow('Cone Detection', img)   
            if cv2.waitKey(1) & 0xFF == ord(' '):
               break         
            
            print("cone in center: ", cone_in_center)  
            if cone_in_center:
                #cv2.putText(img, "Arrow in centre", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print("Depth from more than 3:", depth)
                return cone_in_center, "Not Available", cone_center, depth
            else:
                return False, "Not available", None, 2.5
            
            
            
            print("imshow")
            cv2.imshow('YOLO V8 Detection', img)            
            #return arrow_detected, "Not Available", arrow_center, depth
        # Stop streaming
        self.pipeline.stop()
        cv2.destroyAllWindows()
        #return False, "Not available", None, 0.0
    
    def main(self):
        gps_data_pub.publish(self.gpscalled) 
        
        if self.count_arrow == self.arrow_numbers: #change before competition  #if we take intervention, change this number acc to your needs
            print("Im in if block of main")
            if(not self.turn and not self.ret) or (self.init):
                self.search()
                if(self.searchcalled):
                    self.process_dict()

            if not self.turn:
                self.ret,direction,pix,self.distance  = self.cone_model()
                if self.ret == True:
                    print("Cone detected")  
                    print("self.cone_distance:",self.distance)

                else:
                    print("Still searching")
                    self.ret = False
                self.move_straight()
            else:   #if self.turn is true, everything stops and rover does only turning
                print("Im going into rotate block in main")
                if(self.direction=="left"):
                    print("rotating left")
                    self.rotate(1)
                else:
                    print("rotating right")
                    self.rotate(-1)


        else:
            print("Im in else block of main")
            if(not self.turn and not self.ret) or (self.init):
                self.search()
                print("self.distance:",self.distance)
                if(self.searchcalled):
                    self.process_dict()
            if(not self.turn):      #this is there because detection need not happen when turning 
                # note that self.turn is made true once in process_dict(), so we put this if condition again.

                self.ret,direction,pix,self.distance=self.arrowdetectmorethan3()
                if self.distance<3.0:
                    ret, depth_frame, color_frame, depth = self.get_frame()
                    ret,direction,pix,distance=self.arrowdetectlessthan3(color_frame,depth)
                    # if not self.ret:
                    #     self.ret,self.direction,pix,self.distance=self.arrowdetectmorethan3()
                    if ret==True:
                        self.ret=ret
                        self.direction=direction
                        self.distance=distance

                if (self.ret):
                    print("arrow detected at distance: "+str(self.distance))
                    print("Direction: " + self.direction)
                else:
                    print("Trying to detect arrow...")
                self.move_straight()

            else:   #if self.turn is true, everything stops and rover does only turning
                print("Im going into rotate block in main")
                if(self.direction=="left"):
                    print("rotating left")
                    self.rotate(1)
                else:
                    print("rotating right")
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
        self.state = msg.data

    def yaw_callback(self,msg):
        if (self.initial_drift_angle==0):   #only for 1st time
            self.initial_drift_angle=self.z_angle
        self.z_angle = msg.data

    def enc_callback(self,msg):
        self.enc_data = msg.data

    def gps_callback(self,msg):
        if(msg.latitude and  msg.longitude):
            self.current_latitude=msg.latitude
            self.current_longitude=msg.longitude
            file_object=open("coordinates.txt","w")
            file_object.write("latitude :%f, longitude :%f",msg.latitude,msg.longitude)
            file_object.close()

    def move_straight(self):
        msg = WheelRpm()
        msg.hb=False
        
        #Aadit's p controller
        if (abs(self.initial_drift_angle-self.z_angle) > 5):    #only if large drift (of 10 degrees) is there, correct it.
            while (abs(self.initial_drift_angle-self.z_angle) > 5):    
                #while it doesn't come back to near the proper orientation, keep giving omega

                msg.omega=int(127+self.kp_straight_rot*(self.initial_drift_angle-self.z_angle))
                msg.omega+=10*int((self.initial_drift_angle-self.z_angle)/abs(self.initial_drift_angle-self.z_angle))   #in case it can't rotate fast    # c*x/abs(x) is basically c*(sign of x)
                
                #capping velocity
                if(msg.omega<95):
                    msg.omega = 95
                elif(msg.omega>159):
                    msg.omega = 159
                msg.vel=127
                print("correcting drift with omega =", msg.omega)
                wheelrpm_pub.publish(msg)
                rate.sleep()
            rospy.sleep(1)
        
        msg.omega=127

        if(self.init or self.searchcalled):
            print("move_straight() is being ignored due to search().")
            msg.vel=127
            wheelrpm_pub.publish(msg)
        elif(self.ret):
            if(abs(self.circle_dist-self.distance)>self.dist_thresh):
                if self.distance!=0.0 and self.distance!=2.5:
                    msg.vel=max(102,int(127+self.kp*(self.circle_dist-self.distance)))
                    print("Moving straight. ",(self.circle_dist-self.distance))
                    wheelrpm_pub.publish(msg)
                else:
                    msg.vel=90
                    wheelrpm_pub.publish(msg)

            else:
                msg.vel=127
                msg.omega=127
                wheelrpm_pub.publish(msg)
                print("Stopped going Straight")
                self.gpscalled = 1
                gps_data_pub.publish(self.gpscalled)
                for i in range(100):
                    gps_data_pub.publish(self.gpscalled)     
                    rate.sleep()                               # 10s   # Competition rules say 10s                    
                if self.count_arrow <= self.arrow_numbers:
                    #rospy.sleep(10)
                    #self.latlong[0].append(msg.latitude)
                    #self.latlong[1].append(msg.longitude)
                    self.gpscalled = 0
                    print()
                    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                    print("lat:",self.current_latitude) 
                    print("long:",self.current_longitude)
                    print()
                    self.turn = True
                    self.rotate_angle = 90
                    self.initial_yaw=self.z_angle
                    self.count_arrow += 1
                else:
                    self.v1_competition()

#                self.write_coordinates()

        else:
            print("Forward")
            msg.vel = 102
            wheelrpm_pub.publish(msg)
            self.turn=False

    def rotate(self,dir):
        print("Rotation has been called")
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
        print("Rotation angle:",self.rotate_angle)
        print("Initial Yaw:", self.initial_yaw)
        print("Current z angle", self.z_angle)
        error = self.rotate_angle-abs(diff)
        print("error=", error)
        #if self.direction == -1:
         #   self.rotate_angle = self.rotate_angle +2
        if (abs(error)>=0.5*self.angle_thresh):
            msg.omega=127+(dir*40)
            msg.vel=102
            print("Calling Rotate, printing Z angle below")
            print(error)
            wheelrpm_pub.publish(msg)
        else:
            msg.omega=127
            msg.vel = 127
            wheelrpm_pub.publish(msg)
            # self.z_angle=0

            self.initial_drift_angle=self.z_angle
            print("****ROTATE DONE*****")
            #self.distance = 10.0
            self.start_time = time.time()-10
            self.turn=False
            self.direction = "Not Available"
            rospy.sleep(2)
        
        #self.manjari = False

    def stop_run(self):
        msg_stop=WheelRpm()
        msg_stop.hb=False
        msg_stop.vel=msg_stop.omega=127
        wheelrpm_pub.publish(msg_stop)
#        self.bag.close()
        
    def v1_competition(self):
        msg_stop=WheelRpm()
        msg_stop.hb=False
        msg_stop.vel=msg_stop.omega=127
        wheelrpm_pub.publish(msg_stop)
        print("Course completed(hopefully)")
        while not rospy.is_shutdown():
            if(self.state==True):
                print("Press 'A' to go to joystick mode.")
                rate.sleep()

            else:
                print("Rover in Joystick mode")
                rate.sleep()


if __name__ == '__main__':
    try:
        rospy.init_node("arrowdetectmorethan3")
        rate=rospy.Rate(10)
#    i=int(input("Enter test number: "))
        wheelrpm_pub=rospy.Publisher('motion',WheelRpm,queue_size=10)
        gps_data_pub = rospy.Publisher('gps_bool',std_msgs.Int8,queue_size=10)
        run=auto()
        run.spin()
    except KeyboardInterrupt:
    # quit
        sys.exit()

