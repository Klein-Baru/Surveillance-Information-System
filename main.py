import cv2
from simple_face_recognition import SimpleFaceRecognition

# start webcam
cap = cv2.VideoCapture(0)

# Encode faces from folder
sfr = SimpleFaceRecognition()
sfr.load_encoding_images("images/")

# show frame by frame
while True:

    ret , frame = cap.read()
    # Detect Faces
    face_place,face_names = sfr.detect_known_faces(frame)
    for face_pl , name in zip(face_place, face_names):
        y1 , x2 , y2 , x1 = face_pl[0], face_pl[1], face_pl[2], face_pl[3]

        if name == "Intruder":
            cv2.putText(frame, name, (x1,y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (7, 7, 222), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (7, 7, 222), 4)

        else:
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (222, 126, 7), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (222, 126, 7), 4)

    cv2.imshow("Surveillance", frame)

    key = cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()