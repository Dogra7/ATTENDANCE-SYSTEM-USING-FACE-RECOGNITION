import os
import cv2
import numpy as np
import face_recognition

# Path to the folder containing test images
test_folder = "test_images"

# Load known faces (same as in your Flask app)
folder_path = "faces"
known_face_encodings = []
known_face_names = []
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image = face_recognition.load_image_file(os.path.join(folder_path, filename))
        face_encodings = face_recognition.face_encodings(image)
        if len(face_encodings) > 0:
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])

# Initialize variables for evaluation
total_faces = 0
correct_predictions = 0

# Iterate through test images
for filename in os.listdir(test_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        test_image = face_recognition.load_image_file(os.path.join(test_folder, filename))
        test_face_locations = face_recognition.face_locations(test_image)
        test_face_encodings = face_recognition.face_encodings(test_image, test_face_locations)

        # Iterate through faces in the test image
        for test_face_encoding in test_face_encodings:
            total_faces += 1
            matches = face_recognition.compare_faces(known_face_encodings, test_face_encoding)
            if True in matches:
                correct_predictions += 1

# Calculate accuracy
accuracy = (correct_predictions / total_faces) * 100
print("Total faces:", total_faces)
print("Correct predictions:", correct_predictions)
print("Accuracy:", accuracy, "%")