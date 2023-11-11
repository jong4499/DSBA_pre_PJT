import cv2
import os

src_path = 'c:/Users/admin/Desktop/ttp_3.mp4'
save_path = 'c:/Users/admin/Desktop/vid_results'
count = 0


video = cv2.VideoCapture(src_path) 

if not video.isOpened():
    print("Could not Open :", src_path)
    exit(0)


length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)


while(video.isOpened()):
    ret, image = video.read()
    if(int(video.get(1)) % fps == 0):
        cv2.imwrite(save_path + "/ttp3_%d.jpg" % count, image)
        print('Saved frame number :', str(int(video.get(1))))
        count += 1
        
video.release()