import face_recognition
import cv2
import os
import numpy as np

def load_known_images(folder_path):
	
	known_faces, known_names = [], []

	for path in os.listdir(folder_path):
		img = face_recognition.load_image_file(os.path.join(folder_path, path))
		known_faces.append(face_recognition.face_encodings(img)[0])
		known_names.append(path.split('.')[0])

	return known_faces, known_names

known_faces, known_names = load_known_images('./__img__/known')

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
	raise IOError("Cannot open webcam")

while True:
	ret, frame = cap.read()

	rgb_frame = frame[:, :, ::-1]

	# frame_rec = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

	face_locations = face_recognition.face_locations(rgb_frame)
	face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

	for (top, right, bottom, left), encode_face in zip(face_locations, face_encodings):

		matches = face_recognition.compare_faces(known_faces, encode_face)

		distances = face_recognition.face_distance(known_faces, encode_face)

		text = 'Unk'
		color = (0,0,255)

		best_index = np.argmin(distances)
		if matches[best_index]:
			text = known_names[best_index]
			color = (0,255,0)

		cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
		cv2.putText(frame, text, (left, bottom-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

	cv2.imshow('Input', frame)

	c = cv2.waitKey(1)

	# ascii number of ESC key
	if c == 27:
		break

cap.release()
cv2.destroyAllWindows()
