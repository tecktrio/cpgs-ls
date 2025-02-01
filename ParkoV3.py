import datetime
import math
import time
import cv2
import json
import numpy as np
import pickle
import easyocr
import threading
import socket
# from picamera2 import Picamera2


points = []
corner = []



class ParkoV3:
    def __init__(self):
        '''Initialize all the variables or data on boot'''
  
        # Device operation Mode
        # 0 - live mode
        # 1 - auto_calibrate mode
        # 2 - manual_calibrate mode
        # default mode
        # self.picam2 = Picamera2()
        # self.picam2.start()
        self.picam2 = cv2.VideoCapture(0)  # Change to the appropriate camera index if needed
        self.mode = 1
        # Change this to False on production to reduce processing power
        self.DEBUG = True
        # Getting the Device Ip and configring the port for Socket connection
        HostName = socket.gethostname()
        # self.serverIP = socket.gethostbyname(HostName)
        self.serverIP = ""
        self.serverPort = ""
        # Format of data to send to the server at times when there is any update in the status
        self.slotData = {
            "timestamp":"",
            "slot":"",
            "status":"",
            "licence_number":""
        }
        # Make sure this doesn't change , use the key where you want to set the status
        self.slotStatus = {
            "occupied":200,
            "vaccant":201,
            "reserved":202,
            "not_confirmed":203
        }
        # Device status 
        self.status = 'Active'
        # Waiting time to confirm whether the vehicle is parked or not.
        # The program will wait for this much of time when a status change is dectected. 
        # It again reads the status to confirm the change in status
        self.confirmationIntervalInSecond = 1
        # Update this value based on the slot that the device wants to manage
        self.NumberOfSlotsManaged = 3
        # Camera port is set while object is created, change it over there if its neccessary
        # self.camera_port = camera_port
        # Store the last available slots
        self.lastAvailableSlots = []
        self.VaccantSlots = []
        self.OccupiedSlots = []
        self.listOfLicenceNumbers = []
        # Status change or sensing of vehicle happens based on this trigger value
        self.TriggerVehicleAt = 50
        self.scanTimeInSec = 5
        # Create a object for the camera
        # self.camera = cv2.VideoCapture(self.camera_port)

        # self.frame = self.picam2.capture_array()
        # Config the DEBUG window size
        # self.camera.set(3, 640)
        # self.camera.set(4, 420)
        # Store all the 4 coordinates of each slot on calibration time
        self.coordinate_data = []
        # Calibration configuration
        self.calibrateBorderColor = (0, 255, 0)
        self.calibrateBorderthinkness = 3
        # List to store the coordinates of each slot on normal run
        self.poslist = []
        # Default values for filtering each frame, it will update based on the light intensity
        self.CameraFilterThresh = 0
        self.CameraFilterMaximumThresh = 0
        # Default thresh boost value, change it carefully
        self.BoostThreshAT = 2

        self.CameraFilterThreshOnCalibrate = 0
        self.CameraFilterMaximumThreshOnCalibrate  = 0
        # loading the coordinates of the parking area from the coordinate file

        with open('coordinates','rb')as data:
            self.poslist = pickle.load(data)
        

    def load_next_frame(self):
        '''
        Frame will be assigned with a latest camera frame or image
        '''
        # if not self.camera.isOpened():
        #     print('Camera not found')x
        #     return False
        # else:
        ret, self.frame = self.picam2.read()
        # self.frame = self.picam2.capture_array()
        return True
    
    def current_timestamp(self):
        '''
        Returns the current timestamp
        '''
        return datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    
    def update_configurations(self):
        '''
        Reads the values for filter from the config file and update it in the program.
        '''
        with open("config.json","rb") as configurations:
            config = json.load(configurations)
            self.CameraFilterThresh = config["threshold"]
            self.serverIP = config["server_ip"]
            self.serverPort = config["server_port"]
            self.DEBUG = config["DEBUG"]
            self.CameraFilterMaximumThresh = 255
            self.CameraFilterThreshOnCalibrate = config["threshold"]
            self.CameraFilterMaximumThreshOnCalibrate = 255
        return True
 
    def scan_license_number(self,slotIndex):
        ''''
        Input - Slot index
        Output - Licence Number of the vehicle parked
        '''
        self.listOfLicenceNumbers = []
        slotImage = cv2.imread(f'SlotImages/{slotIndex}.jpg')
        reader = easyocr.Reader(model_storage_directory = '/Users/apple/Desktop/ParkoV1/ParkoV2/LanguageModels', lang_list = ['en'])
        result = reader.readtext(slotImage)
        data = [entry[1] for entry in result]
        for text in data:
            # print("Found : ",str(text).replace(" ",''))
            if len(text) != 0 and len(text) > 4:
            # if len(text) != 0:
                self.listOfLicenceNumbers.append({'slot_id':slotIndex,'Number':text}) 
        return {'status':True,'data':self.listOfLicenceNumbers}

    def get_vehicle_percentage(self,thresholded_image, slot_coordinates):
            '''
            Return the percentage of object in the slot.
            '''
            pts = np.array(slot_coordinates, np.int32)
            # Find the bounding box of the polygon
            x, y, w, h = cv2.boundingRect(pts)
            cropped_image = thresholded_image[y:y+h, x:x+w]
            # if self.DEBUG:
            #     cv2.imshow(f'{slot_coordinates}',cropped_image)
            zero_count = cv2.countNonZero(cropped_image)
            return zero_count
    
    def save_slot_image(self, slotIndex, coordinates):
        '''
        Saves the image of slot into the folder SlotImages
        '''
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(coordinates)
        # Crop the original frame using the bounding box coordinates
        cropped_image = self.frame[y:y+h, x:x+w]
        cv2.imwrite(f'SlotImages/{slotIndex}.jpg',cropped_image)
    
        return True
    
    def scan_slots(self):
        '''
        Dectect the parking slot
        Save the coordinates
        '''
        timer = 0
        self.load_next_frame()
        timer += 1
        # converts to gray scale 
        frameInGray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        # Add some blur to make the dectection more clear
        frameInGrayAndBlur = cv2.GaussianBlur(frameInGray, (3, 3), 2)
        # Add some threshold to make the area of interest differentiate from its background
        self.update_configurations()
        _,ThreshHoldedFrame  = cv2.threshold(frameInGrayAndBlur, self.CameraFilterThresh, self.CameraFilterMaximumThresh, cv2.THRESH_BINARY_INV)
        imgmedian = cv2.medianBlur(ThreshHoldedFrame, 5)
        kernal = np.ones((3, 3), np.uint8)
        # Increase the area of interest width to make it easy to identify
        imgdilate = cv2.dilate(imgmedian, kernel=kernal, iterations=self.BoostThreshAT)
        self.VaccantSlots = []
        self.OccupiedSlots = []
        for slotIndex, pos in enumerate(self.poslist):
            # Define the coordinates of the contour
            SlotCoordinates = np.array([[pos[0][0], pos[0][1]], [pos[1][0], pos[1][1]], [pos[2][0], pos[2][1]], [pos[3][0], pos[3][1]]])
            contours = [SlotCoordinates]
            vehiclePercentage = self.get_vehicle_percentage(imgdilate, SlotCoordinates)
            if vehiclePercentage < self.TriggerVehicleAt:
                color = (0,255,0)
                self.VaccantSlots.append(slotIndex)
            else:
                self.OccupiedSlots.append(slotIndex)
                color = (255,0,255)
            # DEBUG
            cv2.drawContours(self.frame, contours, 0, color, 2)
        # # Choose the font type, scale, color, and thicknessqqqq
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)  # white color in BGR
        thickness = 2
        # Draw the text on the image
        cv2.putText(self.frame, str(len(self.VaccantSlots)), (50, 50), font, font_scale, font_color, thickness)
                
        # if timer == self.scanTimeInSec:
        #         break
        # Display the image with rectangles drawn on matching contours
        if self.DEBUG:
            cv2.imshow('Dectecting Parking spaces', self.frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.camera.release()
        return True , self.VaccantSlots, self.OccupiedSlots
    
    def manual_calibrate(self):
        coordinates = []
        global points, corner
        # Define the mouse callback function
        def get_mouse_coordinates(event, x, y):
            global points, corner
            if event == cv2.EVENT_LBUTTONDOWN:  # Check for left mouse button click
                points.append((x, y))

                if len(points)%5 == 0:
                    # print('one slot ready')
                    coordinates.append(points)

                    with open('coordinates','wb') as coordinate:
                        pickle.dump(coordinates, coordinate)
                        print('Updated New coordinates to DB')
                    points = []

        # Display the image
        while True:
            # cv2.imshow('frame', self.frame)
            self.load_next_frame()
            if len(points) > 1:
                    for index in range (0,len(points)-1):
                        cv2.line(self.frame, points[index], points[index+1], (0, 255, 0), 2)  # Draw line
            for slot_coordinates in coordinates:
                    for index in range (0,len(slot_coordinates)-1):
                        cv2.line(self.frame, slot_coordinates[index], slot_coordinates[index+1], (0, 255, 0), 2)  # Draw line

            cv2.setMouseCallback('frame', get_mouse_coordinates)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
 
    def auto_calibrate(self):
        '''
        1. Find all possible parking slots from the frame.
        2. Store the coordinates of founded slots to the coordinates file.
        3. It takes average 5 second to complete this task.
        summary
        -------
        Your Db will get updated with new coordinates if this function is called.
        '''
        timer = 0
        while self.load_next_frame():
            # time.sleep(1)
            timer += 1
            # converts to gray scale 
            frameInGray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            # Add some blur to make the dectection more clear
            frameInGrayAndBlur = cv2.GaussianBlur(frameInGray, (3, 3), 2)
            # Add some threshold to make the area of interest differentiate from its background
            self.update_configurations()
            _,ThreshHoldedFrame  = cv2.threshold(frameInGrayAndBlur, self.CameraFilterThreshOnCalibrate, self.CameraFilterMaximumThreshOnCalibrate, cv2.THRESH_BINARY_INV)
            imgmedian = cv2.medianBlur(ThreshHoldedFrame, 3 )
            kernal = np.ones((3, 3), np.uint8)
            # Increase the area of interest width to make it easy to identify
            imgdilate = cv2.dilate(imgmedian, kernel=kernal, iterations=self.BoostThreshAT)
            # Find contours
            contours, _ = cv2.findContours(imgdilate, cv2.INTER_AREA, cv2.CHAIN_APPROX_SIMPLE)
            # Iterate through contours
            self.coordinate_data = []
            for slotIndex, contour in enumerate(contours):
                # Calculate the area and perimeter of the contour
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                # # Set area and perimeter thresholds based on your specific requirements
                area_threshold = 500
                perimeter_threshold = 200
                # # Check if the contour matches the rectangle criteria
                if area > area_threshold and perimeter > perimeter_threshold:
                #     # Calculate the minimum area bounding rectangle
                    epsilon = 0.02 * perimeter  # Adjust the epsilon value as needed
                    approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
                    x, y, w, h = cv2.boundingRect(approx_polygon)
                    if len(approx_polygon) == 4 and h > w :
                        corners = [tuple(coord[0]) for coord in approx_polygon]
                        # Calculate vectors representing two sides of the rectangle
                        vector1 = (corners[1][0] - corners[0][0], corners[1][1] - corners[0][1])
                        vector2 = (corners[2][0] - corners[1][0], corners[2][1] - corners[1][1])
                        # Function to calculate angle between two vectors
                        dot_product = sum(a * b for a, b in zip(vector1, vector2))
                        magnitude1 = math.sqrt(sum(a * a for a in vector1))
                        magnitude2 = math.sqrt(sum(b * b for b in vector2))
                        cosine_angle = dot_product / (magnitude1 * magnitude2)
                        angle = math.degrees(math.acos(cosine_angle))
                        if angle >70 and angle <110:
                            self.coordinate_data.append(corners)
                            cv2.drawContours(self.frame, [contour], 0, self.calibrateBorderColor, self.calibrateBorderthinkness)

            if timer == self.scanTimeInSec:
                with open('coordinates','wb') as coordinates:
                    pickle.dump(self.coordinate_data, coordinates)
                    print('Updated New coordinates to DB')
                    break
            # Display the image with rectangles drawn on matching contours
            if self.DEBUG:
                cv2.imshow('Frame in Gray And Blured', frameInGrayAndBlur)
                cv2.imshow('Frame', self.frame)
                cv2.imshow('Thresholded Frame', ThreshHoldedFrame)
                cv2.imshow('Dilated Frame', imgdilate)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                # self.camera.release()          
                with open('coordinates','wb') as coordinates:
                    pickle.dump(self.coordinate_data, coordinates)
                    print('Updated New coordinates to DB')
                    break
        return True
   
    def network_handler(self, timestamp = None, slot = None, status = None, licenseNumber = None):
        '''
        Sends the slotData to the server in UDP protocol
        '''
        self.slotData["timestamp"] = timestamp
        self.slotData["slot"] = slot
        self.slotData["status"] = status
        self.slotData["licence_number"] = licenseNumber
        bytesToSend  = str(self.slotData).encode('utf-8')
        serverSocketAddress   = (self.serverIP, self.serverPort)
        bufferSize          = 1024
        print(self.slotData)
        try:
            UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
            UDPClientSocket.sendto(bytesToSend, serverSocketAddress)
        except Exception as e:
            print(e)

    def start(self):
        ''''
        There are two modes
        1. Live mode
        ------------
        => Start
        => Get Frame
        => Checks for status change in any of the managing slots
        => If a change is found , Confirm the change
        => Dectect the details like licence number
        => Send this data to the server. This data includes the requesting time,  slot unique id, license number
        => End

        2. Auto Calibrate mode
        -----------------
        => Start
        => Get Frame
        => Find all the possible parking slots from the frame
        => Update the coordinate in every calibratetimeinsec.
        => Save the coordinates in coordenates file
        => End

        3. Manual Calibrate mode
        -----------------------
        => start
        => Mark the slots on live video feed
        => Save the slot marks
        => End
        '''

        

        # Mode 1
        if self.mode == 0:
            timer = 0
            _, CurrentAvailableSlots , OccupiedSlots = parkov3.scan_slots()
            self.lastAvailableSlots = CurrentAvailableSlots
            while True:
                timer += 1
                # Confirm the parking of vehicle
                _, CurrentAvailableSlots , OccupiedSlots = parkov3.scan_slots()
                # Checks for changes in parking slot
                if len(CurrentAvailableSlots) != len(self.lastAvailableSlots):
                    # print('Movement sensed, Waiting for confirmation...')
                    # self.network_handler(timestamp = self.current_timestamp(), status  = self.slotStatus["not_confirmed"])
                    # Waiting few seconds for the vehicle to park
                    time.sleep(self.confirmationIntervalInSecond)
                    # Checking again to confirm the existence of vehicle in the spot
                    _, CurrentAvailableSlots , OccupiedSlots = parkov3.scan_slots()
                    # Runs when the vehicle is parked 
                    if len(CurrentAvailableSlots) < len(self.lastAvailableSlots):
                        # print(f'Parking Confirmed for slot {self.getSlotId(OccupiedSlots[-1])}')
                        print('processing...')
                        self.lastAvailableSlots = CurrentAvailableSlots
                        # for slotIndex2, pos in enumerate(self.poslist):
                        pos = self.poslist[OccupiedSlots[-1]]
                        # Define the coordinates of the contour
                        SlotCoordinates = np.array([[pos[0][0], pos[0][1]], [pos[1][0], pos[1][1]], [pos[2][0], pos[2][1]], [pos[3][0], pos[3][1]]])
                        # Only Update when the car is parked for a specific time
                        # print(f'Saving Image of slot {self.getSlotId(OccupiedSlots[-1])}')
                        self.save_slot_image(OccupiedSlots[-1], SlotCoordinates)
                        for slotIndex in OccupiedSlots:
                            def getLicenceNumber():
                                waitToScanFor = 10
                                timer = 0
                                while True:
                                    timer += 1
                                    if timer == waitToScanFor:
                                        break
                                    time.sleep(1)
                                # print('Slot Image Send for Dectection')
                                LicenceNumber = parkov3.scan_license_number(slotIndex)
                                # print("Vehicle Details",LicenceNumber)
                                self.network_handler(timestamp = self.current_timestamp(), status  = self.slotStatus["occupied"], licenseNumber = LicenceNumber['data'])
                            thread1 = threading.Thread(target=getLicenceNumber)
                            thread1.start()
                            # LicenceNumber = parkov3.scan_license_number(slotIndex)
                    # Runs when the vehicle is unparked
                    elif len(CurrentAvailableSlots) > len(self.lastAvailableSlots):
                        # print(f'UnParking Confirmed on slot {self.getSlotId(CurrentAvailableSlots[-1])}')
                        self.network_handler(timestamp = self.current_timestamp(), status  = self.slotStatus["vaccant"])
                        self.lastAvailableSlots = CurrentAvailableSlots
                    # Runs if it was a false dectection
                    else:
                        _, CurrentAvailableSlots , OccupiedSlots = parkov3.scan_slots()
                        # self.lastAvailableSlots = CurrentAvailableSlots
                        print('Confirmation Failed, Invalid sense')

                if timer == 100:
                    timer = 0
                    # print(f"{self.status} => {self.current_timestamp()}")
        # Mode 2
        elif self.mode == 1:
            while True:
                parkov3.auto_calibrate()
        # Mode 3
        elif self.mode == 2:
            parkov3.manual_calibrate()

if __name__ == '__main__':
    print('starting....')
    try:
        parkov3 = ParkoV3()
        parkov3.start()
    except Exception as error:
        # parkov3.network_handler(f"Need Troubleshooting for the device {socket.gethostbyname(socket.gethostname())}")
        print('Something went wrong. Sending the Error details to the Parko')
        print('---------  Error ----------')
        print(f"{error}")
  