from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import os
import face_recognition
from face_recognition.api import load_image_file
import glob
import csv
import clx.xms
import requests
import clx.xms



from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

def get_name(frame):
        faces_encodings = []
        faces_names = []
        _path = os.path.join('dataset/' + 'images/')

        list_of_files = [i for i in glob.glob(_path+'*.jpg')]

        number_files = len(list_of_files)
        names = list_of_files.copy()                


        for i in range(number_files):
            globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_of_files[i])
            globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]
            faces_encodings.append(globals()['image_encoding_{}'.format(i)])

            names[i] = names[i].replace(_path, "")  
            faces_names.append(names[i]) 

        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True

        
        rgb_small_frame = frame[:, :, ::-1]
        if process_this_frame:
            face_locations = face_recognition.face_locations( rgb_small_frame)
            face_encodings = face_recognition.face_encodings( rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces (faces_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance( faces_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = faces_names[best_match_index]
                face_names.append(name.split(".")[0][-1])

        return face_names 


def detect_and_predict_mask(frame, faceNet, maskNet):
        # grab the dimensions of the frame and then construct a blob
        # from it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
            (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        faceNet.setInput(blob)
        detections = faceNet.forward()

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            # for faster inference we'll make batch predictions on all
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)

        # return a 2-tuple of the face locations and their corresponding
        # locations
        return (locs, preds)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('home.html')

@app.route('/video_feed', methods=['GET'] )
def video_feed():
    prototxtPath = r"face_detector/deploy.prototxt"
    weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    maskNet = load_model("mask_detector.model")

    # initialize the video stream
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    count = 0
    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=600)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        count = count + 1
        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            result = label
            
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        cv2.moveWindow("Frame", 200, 500)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if count == 180 or key == ord("q"):
            if result == "No Mask":
                l=get_name(frame)
                print(l)
        
                with open('dataset/emp.csv') as csv_file:
                        csv_reader= csv.reader(csv_file, delimiter=',')
                        for row in csv_reader:
                            if row[0] == l[0]:
                                mobile = row[8]
                                client = clx.xms.Client(service_plan_id='5e51fff866b24a0fab3f1ac9436204ac',token='2b02c0df412349faa9157af30b294214')
                                create = clx.xms.api.MtBatchTextSmsCreate()
                                create.sender = '+447537454491'
                                create.recipients = {'+91'+ mobile}
                                message = 'Message sent to: '+ 'XXXXXX' + mobile[6:]
                                print('Message sent to '+ row[9])
                                create.body = 'Hello ' + row[9] + ', Please wear mask, Lets fight covid together.' 
                                client.create_batch(create)
            else:
                message = "Great that you are following the covid guidelines"
                
            break

    
    cv2.destroyAllWindows()
    vs.stop()

    return render_template('check.html', prediction=message)

if __name__ == '_main_':
    app.run(debug=True)