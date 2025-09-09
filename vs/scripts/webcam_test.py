import cv2

def open_webcam():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("webcam not opened")
        return
    
    print("webcam started!!! wohoo!!!")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("failed to grab frame")
            break
        
        cv2.imshow("webcam test",frame)
        
        # we use the key q to break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    open_webcam()
            