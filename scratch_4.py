import cv2

#using the cascade for face recognition
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def find_and_blur(bw, color):
    #detect all faces in video capture
    faces = cascade.detectMultiScale(bw, 1.1, 4)
    #for each face, we apply gaussian blur
    for (x, y, w, h) in faces:
        #draw a rectangle surrounding the face,optionally
        #cv2.rectangle(color, (x, y), (x + w, y + h), (255, 255, 0), 5)
        #get the portion of the image containing the face
        area_color = color[y:y + h, x:x + w]
        # blur the colored image
        blur = cv2.GaussianBlur(area_color, (55, 55), 0)
        # insert the blurred part back into the image
        color[y:y + h, x:x + w] = blur

        # return the blurred image
    return color
#capture video
video_capture = cv2.VideoCapture(0)

while True:
    # get last recorded frame
    _, color = video_capture.read()
    # transform color into grayscale
    bw = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    # detect the face and blur it
    blur = find_and_blur(bw, color)
    # display output
    cv2.imshow('Video', blur)
    # break if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# turn camera off
video_capture.release()
# close camera  window
cv2.destroyAllWindows()