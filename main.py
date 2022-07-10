from threading import Thread
import cv2
from tensorflow import keras
import numpy as np
#constants
FACE_DETECTION_MODEL = 'haarcascade_frontalface_default.xml'
SMILE_DETECTION_MODEL = 'smile_v1.h5' #image crop should be 64x64
CAPTURE_NAME = 'SSIP-2022.jpeg'
CROP_IMG_SIZE = (64,64)
def load_smile_model(path):
    model = keras.models.load_model(path)
    return model


def load_face_model(path):
    face_cascade = cv2.CascadeClassifier(path)
    return face_cascade
def do_capture(feed, pic_name):
    thread = Thread(target=cv2.imwrite,args=(pic_name, feed))
    thread.start()
    print("A capture was made!")
def check_numb_of_smiles(predictions):
    num_smiling =0
    num_not_smiling=0
    for pred in predictions:
        if pred[0][1] > 0.5:
            num_smiling+=1
        else:
            num_not_smiling+=1
    return num_smiling

def predict(model,X_test):
  return model.predict(X_test)

def load_camera(camera_port):
    cap = cv2.VideoCapture(camera_port)
    return cap

def detect(camera,face_model,smile_model):
    predictions = []
    captured_photo = False
    while True:
        #read from camera
        _, img = camera.read()
        #convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        faces = face_model.detectMultiScale(gray, 1.3, 4)
        for (x, y, w, h) in faces:
            # get every cropped face
            cropped_image = img[y:h + y, x:w + x]
            cropped_image = cv2.resize(cropped_image, CROP_IMG_SIZE)
            cropped_image = np.array([cropped_image])
            # do a prediction for every face
            prediction = predict(smile_model, cropped_image)
            # do a rectangle around the faces (RED =NO Smile GREEN = Smile)
            if prediction[0][1] > 0.5:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # add prediction to the predictions list
            predictions.append(prediction)
        # determine the no. of smiling faces
        num_smile = check_numb_of_smiles(predictions)
        # reset predictions list
        predictions = []
        # show out to screen
        cv2.imshow('Video', img)
        # print information to console
        print("Number of faces:", len(faces), " and number of smiling faces: {}".format(num_smile))
        # do a screen capture when everyone is smiling
        if len(faces) != 0 and len(faces) == num_smile and captured_photo == False:
            do_capture(img, CAPTURE_NAME)
            captured_photo = True
        # press 'q' to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # press 'r' to reset the trigger for new picture...
        if cv2.waitKey(1) & 0xFF == ord('r'):
            captured_photo = False
            print("Resetting trigger...")
    # when screen is clone we should release the camera...
    camera.release()

if __name__ == '__main__':
    camera = load_camera(0)
    face_model = load_face_model(FACE_DETECTION_MODEL)
    smile_model = load_smile_model(SMILE_DETECTION_MODEL)
    detect(camera=camera,face_model=face_model,smile_model=smile_model)
