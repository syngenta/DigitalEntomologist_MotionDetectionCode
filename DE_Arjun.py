# use this to show bounding box on image
DISPLAY_CONTOURS = True

# Required Libraries
import cv2 as cv
import numpy as np
from datetime import datetime
import json
import logging as log
from skimage.feature import graycomatrix, graycoprops

# Intializing path and variables
log.basicConfig(filename='/var/tmp/cam.log', filemode='w', level=log.INFO, format='[%(asctime)s]- %(message)s', datefmt='%d-%m-%Y %I:%M:%S %p')
log.info("Cam script started..")
with open(f"/etc/entomologist/ento.conf",'r') as file:
    data=json.load(file)

DEVICE_SERIAL_ID = data["device"]["SERIAL_ID"]
BUFFER_IMAGES_PATH = data["device"]["STORAGE_PATH"]

# Creating class for motion detection
class MotionRecorder(object):
    
   
    VID_RESO = (640, 480)

    # video capture : from device
   
    cap = cv.VideoCapture(f"v4l2src device=/dev/video2 ! video/x-raw, width={VID_RESO[0]}, height={VID_RESO[1]}, framerate=60/1, format=(string)UYVY ! decodebin ! videoconvert ! appsink", cv.CAP_GSTREAMER)
    
    # the background Subractors
    subtractor = cv.createBackgroundSubtractorMOG2(detectShadows=False)   

    # FourCC is a 4-byte code used to specify the video codec. The list of available codes can be found in fourcc.org.
    fourcc = cv.VideoWriter_fourcc(*'DIVX')     # for windows
    fps = 60

    IMAGE_COUNTER_LIMIT = 60
    img_counter = 0

    CONTOUR_AREA_LIMIT = 10
    
    temp_img_for_video = []
    temp_img_bbox_for_video = {}
    temp_img_bbox_count = {}
    # red blue colour intensity difference threshold
    RB_threshold = 30

    def _init_(self):
        pass
    
    
    def normalize_image(image,levels):
        norm_img = np.zeros((image.shape[1],image.shape[0]))
        return cv.normalize(image,norm_img,0,50,cv.NORM_MINMAX)


    def process_img(self, frame):
        store = frame

        kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
        frame = cv.GaussianBlur(frame,(5, 5), 0)

        #frame = cv.medianBlur(frame,5, 0)
        mask = MotionRecorder.subtractor.apply(frame)
        gray = mask
        gray = cv.morphologyEx(gray,cv.MORPH_OPEN,cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5)))
        
        imgOut = gray
        imgOut1 = cv.morphologyEx(imgOut, cv.MORPH_OPEN, kernel,iterations = 2)
        imgOut1[imgOut1 == 127] = 0 # 0-> black
        
        imgOut2 = cv.morphologyEx(imgOut1, cv.MORPH_CLOSE, kernel,iterations = 2)
        # remove light variations(gray 127) - consider only Major Movement (white patches 255)    
        imgOut2[imgOut2 == 127] = 0 # 0-> black

        imgOut3 = cv.morphologyEx(imgOut, cv.MORPH_CLOSE, kernel,iterations = 2)
        # remove light variations(gray 127) - consider only Major Movement (white patches 255)    
        imgOut3[imgOut3 == 127] = 0 # 0-> black
        


        contours, _ = cv.findContours(imgOut2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        storee = store
        
        numberofObjects = 0
        
        detections = []

        for cnt in contours:
            x,y,w,h = cv.boundingRect(cnt)
            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            box = np.int0(box)
#             print(box)
            width = int(rect[1][0])
            height = int(rect[1][1])

            src_pts = box.astype("float32")
            dst_pts = np.array([[0, height-1],
                                [0, 0],
                                [width-1, 0],
                                [width-1, height-1]], dtype="float32")

            M = cv.getPerspectiveTransform(src_pts, dst_pts)

            warped = cv.warpPerspective(storee, M, (width, height))
            try:
                warped_img = warped
                warped_img = cv.resize(warped_img,(50,50))
#                 
                image = cv.cvtColor(warped_img, cv.COLOR_BGR2GRAY)
                
    
                image = MotionRecorder.normalize_image(image,levels=50)
            
                glcm = graycomatrix(image,distances=[2],angles=[np.pi/4],levels=51,symmetric=True,normed=True)

                dis = graycoprops(glcm,'dissimilarity')[0,0]
                
                area = cv.contourArea(cnt)  

                if area > 20 and dis > 1.6:
                    cv.rectangle(store,(x,y),(x+w,y+h),(255,0,0),2)
                    numberofObjects = numberofObjects + 1
                    cv.putText(store,str(numberofObjects), (x,y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv.LINE_8)
                    
                    detections.append([x, y, w, h])

                    
            except:
                pass
        
        cv.putText(store,"Number of objects: " + str(numberofObjects), (10,50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_8)
        

        hasMovement = len(detections) > 0

        return hasMovement, store, detections


    def start_storing_img(self, img):
        '''@vinay
        draw bounding box
        + save bbox json file

        return None'''

        hasMovement, img, bbox = self.process_img(img)
        
        if hasMovement:            
            self.temp_img_for_video.append(img)
            self.temp_img_bbox_for_video[self.img_counter] = bbox
            self.temp_img_bbox_count[self.img_counter] = len(bbox)
            self.img_counter += 1
            if self.img_counter > self.IMAGE_COUNTER_LIMIT:
                self.save_recording()
    
    def save_recording(self):
        if self.img_counter >= 1:   
            now = datetime.now()
            video_name = f'{now.strftime("%d-%m-%Y_%H-%M-%S")}_{DEVICE_SERIAL_ID}.avi'  
            out = cv.VideoWriter(BUFFER_IMAGES_PATH+video_name, self.fourcc, self.fps, (MotionRecorder.VID_RESO[0],MotionRecorder.VID_RESO[1]))
                        
            for image in self.temp_img_for_video : 
                out.write(image)

            # log data
            log.info("Video crealog.info("")ted and saved -> "+video_name)            
            print(video_name)

            # json file name
            json_fname = f'{now.strftime("%d-%m-%Y_%H-%M-%S")}_{DEVICE_SERIAL_ID}.json'            
            json_file = open(BUFFER_IMAGES_PATH+json_fname, 'w')

            # save json bbox
            json.dump(self.temp_img_bbox_for_video, json_file)
            json_file.close()

            # log data
            log.info("Video bbox JSON crealog.info("")ted and saved -> "+json_fname)
            print(json_fname)
            
             # json file name - for bbox count
            json_fname = f'{now.strftime("%d-%m-%Y_%H-%M-%S")}_{DEVICE_SERIAL_ID}_count.json'
            json_file = open(BUFFER_IMAGES_PATH+json_fname, 'w')

            # save json bbox
            json.dump(self.temp_img_bbox_count, json_file)
            json_file.close()

            # log data
            log.info("Video bbox count JSON crealog.info("")ted and saved -> "+json_fname)
            print(json_fname)

            # reset all
            self.temp_img_for_video.clear()
            self.temp_img_bbox_for_video.clear()
            self.temp_img_bbox_count.clear()
            self.img_counter = 0


    def start(self):
        log.info("Cam started functioning")

        #fCount = 0

        while True :
            available, frame = self.cap.read()

            if available:                
                self.start_storing_img(frame)
                
                # check exit
                if cv.waitKey(1) & 0xFF == ord('x'):
                    break
            else:
                if DISPLAY_CONTOURS:
                    print("...Device Unavailable");

    def end(self):        
        self.save_recording()
        self.cap.release()
        cv.destroyAllWindows()


# main
MR = MotionRecorder()
log.info("Object created")
MR.start()
MR.end()
log.info("Script ended")