import cv2
vidcap = cv2.VideoCapture('/data/projects/verification/videa.mp4')
success, image = vidcap.read()
count = 0
while success:
  cv2.imwrite("/data/projects/verification/video_img/%d.jpg" % count, image)
  success, image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1