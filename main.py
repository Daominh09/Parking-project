import os
import cv2
from function import get_parking_space, emty_check

# prepare file
video_path = os.path.join('.', 'Resource', 'parking_1920_1080.mp4')
image_path = os.path.join('.', 'Resource', 'mask_1920_1080.png')

# Check if file exists
if not os.path.isfile(video_path):
    print("Error: Video file not found.")
    exit()
if not os.path.isfile(image_path):
    print("Error: Image file not found. ")
    exit()
    
# Open file
cap = cv2.VideoCapture(video_path) 
mask = cv2.imread(image_path, 0) #Read img in grayscale mode

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spaces = get_parking_space(connected_components)
space_check = [None for i in spaces] 
space_crop = [None for j in spaces]
frame_number = 0
step = 90
ret = True
while ret:
    ret, frame = cap.read()
    if frame_number % step == 0:
        for space_i, space in enumerate(spaces):
            x, y, w, h = space
            space_crop[space_i] = frame[y : y + h, x : x + w, :]
        space_check = emty_check(space_crop)
    for space_i, space in enumerate(spaces):
        x, y, w, h = space
        flag = space_check[space_i]
        if flag == 1:
            frame = cv2.rectangle(frame, (x , y), (x + w , y + h), (0, 255, 0), thickness=2)
        elif flag == 0:
            frame = cv2.rectangle(frame, (x , y), (x + w , y + h), (0, 0, 255), thickness=2)
    cv2.putText(frame, 'Available space {} / {}'.format(str(sum(space_check)), str(len(space_check))), (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    frame_number += 1
    available_space = 0
cap.release()
cv2.destroyAllWindows()