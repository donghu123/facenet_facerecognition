import cv2
video = "http://admin:admin@192.168.1.100:8081/"

cam = cv2.VideoCapture(video)
#cam = cv2.VideoCapture(0)

cv2.namedWindow("camera",1)


while True:
    ret, frame = cam.read()
    #frame = cv2.flip(frame, -1)
  
    
    image1 = cv2.putText(frame,'test', (50,100),
                       cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0 ,0),
                       thickness = 2, lineType = 2)
        
    cv2.imshow('camera',frame)
                       
    
    key = cv2.waitKey(3)
    if key == 27:
       
        print("esc break...")
        break

if key == ord(' '):
    
    num = num + 1
    filename = "frames_%s.jpg" % num
    cv2.imwrite(filename,frame)

