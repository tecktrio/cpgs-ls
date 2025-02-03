
import asyncio
import base64
import json
import math
import pickle
import threading
import time
import base64
import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer
import cv2
import easyocr

from cpgsserver.settings import  IS_PI_CAMERA_SOURCE

DEBUG = True

if IS_PI_CAMERA_SOURCE:
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.start()
else:
    cap = cv2.VideoCapture(1) 


# VIDEO STREMER
async def video_stream():
    if not cap.isOpened():
        if DEBUG:print("Cannot open camera")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            if DEBUG:print("Can't receive frame (stream end?). Exiting ...")
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        encoded_frame = base64.b64encode(frame_bytes).decode('utf-8')
        yield f"data:image/jpeg;base64,{encoded_frame}"
    cap.release()

# VIDEO STREAMER FOR CALIBRATION
async def video_stream_for_calibrate():
    # if not cap.isOpened():
    #     if DEBUG:print("Cannot open camera")
    #     return
    while True:
        # await asyncio.sleep(.1)
        # frame = picam2.capture_array()
        # frame = cap.read()
        if IS_PI_CAMERA_SOURCE:
            frame = cap.capture_array()
        else:
            ret, frame = cap.read()
        # if not ret:
        #     if DEBUG:print("Can't receive frame (stream end?). Exiting ...")

        with open('coordinates','rb')as data:
            for slot_coordinates in pickle.load(data):
                    for index in range (0,len(slot_coordinates)-1):
                        cv2.line(frame, slot_coordinates[index], slot_coordinates[index+1], (0, 255, 0), 2)  # Draw line
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        encoded_frame = base64.b64encode(frame_bytes).decode('utf-8')
            
        readyToSendFrame = f"data:image/jpeg;base64,{encoded_frame}"
        yield readyToSendFrame

# GET ONE FRAME
async def capture():
    if not cap.isOpened():
        if DEBUG:print("Cannot open camera")
        return
    ret, frame = cap.read()
    return frame

# SCAN EACH SPACE/SLOT FOR VEHICLE DECTECTION
async def scan_slots():
        '''
        SCAN the parking slot FOR VEHICLE
        '''
        TriggerVehicleAt = 50
        frameInGray = cv2.cvtColor(await capture(), cv2.COLOR_BGR2GRAY)
        frameInGrayAndBlur = cv2.GaussianBlur(frameInGray, (3, 3), 2)
        with open("config.json","rb") as configurations:
                config = json.load(configurations)
                configuration_data = {
                    "CameraFilterThresh" : config["threshold"],
                    "serverIP" :config["server_ip"],
                    "serverPort" : config["server_port"],
                    "CameraFilterMaximumThresh" : 255,
                    "CameraFilterThreshOnCalibrate" : config["threshold"],
                    "CameraFilterMaximumThreshOnCalibrate" : 255,
                    "BoostThreshAT":2,
                }
        _,ThreshHoldedFrame  = cv2.threshold(frameInGrayAndBlur,
                                                  configuration_data["CameraFilterThreshOnCalibrate"], 
                                                  configuration_data["CameraFilterMaximumThreshOnCalibrate"], 
                                                  cv2.THRESH_BINARY_INV)
        imgmedian = cv2.medianBlur(ThreshHoldedFrame, 5)
        kernal = np.ones((3, 3), np.uint8)
        imgdilate = cv2.dilate(imgmedian, kernel=kernal, iterations=configuration_data["BoostThreshAT"])
        VaccantSlots = []
        OccupiedSlots = []
        poslist =[]
        await asyncio.sleep(.1)
        with open('coordinates','rb')as data:
            poslist = pickle.load(data)
        frame_list_of_cropped_images = []
        for slotIndex, pos in enumerate(poslist):
            SlotCoordinates = np.array([[pos[0][0], pos[0][1]], [pos[1][0], pos[1][1]], [pos[2][0], pos[2][1]], [pos[3][0], pos[3][1]]])
            pts = np.array(SlotCoordinates, np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            cropped_image = imgdilate[y:y+h, x:x+w]
            image_original = await capture()
            cropped_image_original = image_original[y:y+h, x:x+w]
            frame_list_of_cropped_images.append(cropped_image_original)
            zero_count = cv2.countNonZero(cropped_image)
            if zero_count < TriggerVehicleAt:
                VaccantSlots.append(slotIndex)
            else:
                OccupiedSlots.append(slotIndex)
        return True ,VaccantSlots, OccupiedSlots, poslist , frame_list_of_cropped_images
    
# CONSUMER FOR HANDLING ALL REQUESTS FROM CLIENT
class ServerConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.streaming = False
        self.stream_task = None 
        self.coordinates = []
        self.points = [] 

    async def connect(self):
        await self.accept()
        if DEBUG:print('client connected :)')

    async def disconnect(self, close_code):
        if self.streaming:
            if self.stream_task:
                self.stream_task.cancel()
        await super().disconnect(close_code)
        if DEBUG:print('client disconnected :(')

    async def receive(self, text_data=None, bytes_data=None):
        if DEBUG:print('Received Message: ', text_data)
        req = json.loads(text_data)

        # HANDLE LIVESTREAM START REQUESTS
        if req.get("task") == 'start' and not self.streaming:
            self.streaming = True
            self.stream_task = asyncio.create_task(self._stream_frames())
            await asyncio.sleep(.1)

        # HANDLE LIVESTREAM STOP REQUESTS
        elif req.get("task") == 'stop':
            self.stream_task.cancel()

        # HANDLE REQUEST FOR THE MANUAL CALIBRATION FRAMES
        elif req.get("task") == 'get_calibrating_frame':
            self.streaming = True
            self.stream_task = asyncio.create_task(self._stream_calibration_frames())
            await asyncio.sleep(.1)

        # HANDLE REQUEST TO UPDATE THE SPACE COORDINATES IN MANUAL CALIBRATION FRAMES
        elif req.get("task") == 'update_calibrating_frame':
                x = req.get('x')
                y = req.get('y')
                self.points.append((x, y))
                if len(self.points)%5 == 0:
                    self.coordinates.append(self.points)
                    with open('coordinates','wb') as coordinate:
                        pickle.dump(self.coordinates, coordinate)
                    self.points = []

        # HANDLE REQUEST TO RESET THE SPACE COORDINATES IN CPGS DB
        elif req.get("task") == 'reset_calibrating_frame':
            self.coordinates = []
            with open('coordinates','wb') as coordinate:
                        pickle.dump(self.coordinates, coordinate)

        # HANDLE REQUEST FOR AUTOCALIBRATION 
        elif req.get('task') == 'auto_calibrate':
            frameInGray = cv2.cvtColor(await capture(), cv2.COLOR_BGR2GRAY)
            frameInGrayAndBlur = cv2.GaussianBlur(frameInGray, (3, 3), 2)
            with open("config.json","rb") as configurations:
                config = json.load(configurations)
                configuration_data = {
                    "CameraFilterThresh" : config["threshold"],
                    "serverIP" :config["server_ip"],
                    "serverPort" : config["server_port"],
                    "DEBUG" : config["DEBUG"],
                    "CameraFilterMaximumThresh" : 255,
                    "CameraFilterThreshOnCalibrate" : config["threshold"],
                    "CameraFilterMaximumThreshOnCalibrate" : 255,
                    "BoostThreshAT":2,
                }
            _,ThreshHoldedFrame  = cv2.threshold(frameInGrayAndBlur,
                                                  configuration_data["CameraFilterThreshOnCalibrate"], 
                                                  configuration_data["CameraFilterMaximumThreshOnCalibrate"], 
                                                  cv2.THRESH_BINARY_INV)
            imgmedian = cv2.medianBlur(ThreshHoldedFrame, 3 )
            kernal = np.ones((3, 3), np.uint8)
            imgdilate = cv2.dilate(imgmedian, kernel=kernal, iterations=configuration_data["BoostThreshAT"])
            contours, _ = cv2.findContours(imgdilate, cv2.INTER_AREA, cv2.CHAIN_APPROX_SIMPLE)
            self.coordinate_data = []
            for slotIndex, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                area_threshold = 500
                perimeter_threshold = 200
                if area > area_threshold and perimeter > perimeter_threshold:
                    epsilon = 0.02 * perimeter 
                    approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
                    x, y, w, h = cv2.boundingRect(approx_polygon)
                    if len(approx_polygon) == 4 and h > w :
                        corners = [tuple(coord[0]) for coord in approx_polygon]
                        vector1 = (corners[1][0] - corners[0][0], corners[1][1] - corners[0][1])
                        vector2 = (corners[2][0] - corners[1][0], corners[2][1] - corners[1][1])
                        dot_product = sum(a * b for a, b in zip(vector1, vector2))
                        magnitude1 = math.sqrt(sum(a * a for a in vector1))
                        magnitude2 = math.sqrt(sum(b * b for b in vector2))
                        cosine_angle = dot_product / (magnitude1 * magnitude2)
                        angle = math.degrees(math.acos(cosine_angle))
                        if angle >70 and angle <110:
                            self.coordinate_data.append(corners)
            with open('coordinates','wb') as coordinates:
                pickle.dump(self.coordinate_data, coordinates)
        

        # HANDLE REQUEST TO MAKE THE SYSTEM LIVE
        elif req.get('task') == 'live':
            _, CurrentAvailableSlots , OccupiedSlots, poslist, frame_list_of_cropped_images = await scan_slots()
            lastAvailableSlots = CurrentAvailableSlots
            while True:
                await asyncio.sleep(.1)        
                _, CurrentAvailableSlots , OccupiedSlots, poslist, frame_list_of_cropped_images = await scan_slots()
                encoded_spaces = []
                for space in frame_list_of_cropped_images:
                    ret, buffer = cv2.imencode('.jpg', space)
                    frame_bytes = buffer.tobytes()
                    encoded_frame = base64.b64encode(frame_bytes).decode('utf-8')
                    frame = f"data:image/jpeg;base64,{encoded_frame}"
                    encoded_spaces.append(frame)
                await self.send(json.dumps({'fnames': encoded_spaces}))
                if len(CurrentAvailableSlots) != len(lastAvailableSlots):
                    time.sleep(1)
                    _, CurrentAvailableSlots , OccupiedSlots, poslist, frame_list_of_cropped_images = await scan_slots()
                    if len(CurrentAvailableSlots) < len(lastAvailableSlots):
                        for slot in lastAvailableSlots:
                            if slot not in CurrentAvailableSlots:
                                last_parked_slot = slot
                                if DEBUG:print('Parked at space :', slot)
                                await self.send(json.dumps({"slot":slot,"status":"parked"}))

                        lastAvailableSlots = CurrentAvailableSlots
                        pos = poslist[OccupiedSlots[-1]]
                        SlotCoordinates = np.array([[pos[0][0], pos[0][1]], [pos[1][0], pos[1][1]], [pos[2][0], pos[2][1]], [pos[3][0], pos[3][1]]])
                        x, y, w, h = cv2.boundingRect(SlotCoordinates)
                        frame = await capture()
                        cropped_image = frame[y:y+h, x:x+w]

                        
                        def dectect_license_plate(slot):
                            waitToScanFor = 10
                            timer = 0
                            while True:
                                timer += 1
                                if timer == waitToScanFor:
                                    break
                                time.sleep(1)
                            listOfLicenceNumbers = []
                            reader = easyocr.Reader(model_storage_directory = 'LanguageModels', lang_list = ['en'])
                            result = reader.readtext(cropped_image)
                            data = [entry[1] for entry in result]
                            for text in data:
                                if len(text) != 0 and len(text) > 4:
                                    listOfLicenceNumbers.append({'slot_id':slot,'Number':text}) 
                            async def senddata():
                                await self.send(json.dumps({"slot":slot,"license":text}))
                            senddata()
                            if DEBUG:print("Vehicle Details",listOfLicenceNumbers)
                        if DEBUG:print('starting License Plate number dectection',last_parked_slot)
                        tread1 = threading.Thread(target=dectect_license_plate, args=(last_parked_slot,))
                        tread1.start()

                        # self.network_handler(timestamp = self.current_timestamp(), status  = self.slotStatus["occupied"], licenseNumber = LicenceNumber['data'])
                            # thread1 = threading.Thread(target=getLicenceNumber)
                            # thread1.start()
                            # LicenceNumber = scan_license_number(slotIndex)
                            # listOfLicenceNumbers = []
                            # slotImage = cv2.imread(f'SlotImages/{slotIndex}.jpg')
                            # reader = easyocr.Reader(model_storage_directory = 'LanguageModels', lang_list = ['en'])
                            # result = reader.readtext(slotImage)
                            # data = [entry[1] for entry in result]
                            # for text in data:
                            #     # print("Found : ",str(text).replace(" ",''))
                            #     if len(text) != 0 and len(text) > 4:
                            #     # if len(text) != 0:
                            #         listOfLicenceNumbers.append({'slot_id':slotIndex,'Number':text}) 
                            # print({'status':True,'data':listOfLicenceNumbers})


                    # Runs when the vehicle is unparked
                    elif len(CurrentAvailableSlots) > len(lastAvailableSlots):
                        '''
                        Sends the slotData to the server in UDP protocol
                        '''
                        # slotData["timestamp"] = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                        # slotData["slot"] = "slot"
                        # slotData["status"] = slotStatus["vaccant"]
                        # slotData["licence_number"] = "licenseNumber"
                        # bytesToSend  = str(slotData).encode('utf-8')
                        # serverSocketAddress   = ("192.168.1.100", 9090)
                        # bufferSize          = 1024
                        # # print(self.slotData)
                        # try:
                        #     UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
                        #     UDPClientSocket.sendto(bytesToSend, serverSocketAddress)
                        # except Exception as e:
                        #     print(e)

                        for slot in CurrentAvailableSlots:
                            if slot not in lastAvailableSlots:
                                if DEBUG:print('UnParked from space :', slot)
                                await self.send(json.dumps({"slot":slot,"status":"unparked"}))
                        lastAvailableSlots = CurrentAvailableSlots
                        

    async def _stream_frames(self):
        try:
            async for frame in video_stream():
                await asyncio.sleep(0.1) 
                await self.send(frame) 
        except Exception as e:
            if DEBUG:print(f"Error sending frame: {e}")
        finally:
            self.streaming = False 
            self.stream_task = None 

    async def _stream_calibration_frames(self):
        try:
            async for frame in video_stream_for_calibrate():
                await asyncio.sleep(0.1) 
                await self.send(frame) 
        except Exception as e:
            if DEBUG:print(f"Error sending frame: {e}")
        finally:
            self.streaming = False 
            self.stream_task = None 
   