import cv2

#using OpenCV integrated cascade for face recognition
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#reading the image
img = cv2.imread('group.jpg')
#creating a copy on which we will apply the face blur
result_image=img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#detecting faces in the input image
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
#safe explanatory
if len(faces)!=0:
	print("Face detected")
else:
	print("No faces in this photo")
#for each face found, we know the coordinates, width and height of the rectangle surrounding it
for (x, y, w, h) in faces:
	#optionally, we can add a rectangle around the face
    #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 5)
	#creating a sub image of only the face detected
    sub_face = img[y:y + h, x:x + w]
	#applying the opencv built-in gaussian blur
	#the kernel dimensions are given as the second args
	#NOTE: dimensions should be odd and equal; the greater the value of the dimension, the more powerful the blur
    sub_face = cv2.GaussianBlur(sub_face, (13,13), 30)
	#adding the blurred face to the result image
    result_image[y:y + sub_face.shape[0], x:x + sub_face.shape[1]] = sub_face
	#creating a jpg only for the blurred face
    face_file_name = "./face_" + str(y) + ".jpg"
    cv2.imwrite(face_file_name, sub_face)

#creating the new image
cv2.imwrite("./result.png", result_image)


