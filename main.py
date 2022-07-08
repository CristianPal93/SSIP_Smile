import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam.
cap = cv2.VideoCapture(0)
counter = 0
while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces

    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    # Draw the rectangle around each face

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cropped_image = img[y:h+y, x:w+x]
        # if len(cropped_image) > 0:
        #     print("got cropped image")
        #     cv2.imshow("cropped", cropped_image)
        #     cv2.imwrite("{}.jpeg".format(counter), cropped_image)
        #     counter+=1
        #     if counter == 10:
        #         break

    # Display
    cv2.imshow('Video', img)
    print("Number of faces are:",len(faces))
    # Stop if escape key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Release the VideoCapture object
cap.release()
# print(cropped_image)
# cv2.imshow("cropped", cropped_image)