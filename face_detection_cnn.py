import cv2
from mtcnn.mtcnn import MTCNN
detector = MTCNN()

def IITFaceDetection():

    #We can use IP of a webcam to take video stream from
    videoall = cv2.VideoCapture(0)
    while True: 
        #Capture frame-by-frame
        __, image = videoall.read()

        #Using MTCNN to detect faces
        result = detector.detect_faces(image)
        if result != []:
            for person in result:
                bounding_box = person['box']
        
                cv2.rectangle(image,
                              (bounding_box[0], bounding_box[1]),
                              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                              (0,155,255),
                              2)
        
        #display resulting frame
        cv2.imshow('Face Detection App IITD',image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #When everything's done, release capture
    videoall.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    IITFaceDetection()