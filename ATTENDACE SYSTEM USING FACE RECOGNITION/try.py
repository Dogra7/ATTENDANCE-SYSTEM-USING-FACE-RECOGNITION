import face_recognition
import cv2
import psycopg2
from psycopg2 import Error
from datetime import datetime
import os
import numpy as np
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)


def connect_to_database():
    """Establishes a connection to the PostgreSQL database."""
    try:
        connection = psycopg2.connect(
            user="postgres",
            password="deep1234",
            host="localhost",
            port="5432",
            database="ATTENDANCE SYSTEM"
        )
        print("Connected to PostgreSQL database successfully.")
        return connection
    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL database:", error)

def store_attendance_in_database(recognized_face_name):
    """Stores attendance records in the database."""
    try:
        # Establish a connection to the database
        connection = connect_to_database()
        if connection:
            cursor = connection.cursor()

            # Prepare SQL statement to insert attendance record
            insert_query = """INSERT INTO attendance_record (s_name) VALUES (%s)"""
            record_to_insert = (recognized_face_name,)

            # Execute the SQL statement
            cursor.execute(insert_query, record_to_insert)

            # Commit the transaction
            connection.commit()
            print("Attendance record inserted successfully:", recognized_face_name)

    except (Exception, Error) as error:
        print("Error while inserting attendance record:", error)

    finally:
        # Close the database connection
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed.")
# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Define the path to the folder containing images of known faces
folder_path = "faces"

# Initialize arrays to store face encodings and names
known_face_encodings = []
known_face_names = []

# Load known faces
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image = face_recognition.load_image_file(os.path.join(folder_path, filename))
        face_encodings = face_recognition.face_encodings(image)
        if len(face_encodings) > 0:
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])

# Global variable to store recognized face name
recognized_face_name = "Unknown"

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    global recognized_face_name
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        else:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                face_names.append(name)

            # Get the recognized face name
            recognized_face_name = face_names[0] if face_names else "Unknown"
            
            # Draw rectangles and labels on the frame
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                print("Recognized face name:", name)

            # Encode frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Yield frame bytes
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_recognized_face_name')
def get_recognized_face_name():
    global recognized_face_name
    return jsonify({'recognized_face_name': recognized_face_name})

@app.route('/store_recognized_face')
def store_recognized_face():
    store_attendance_in_database(recognized_face_name)
    return jsonify({'message': 'Successfully stored in the database'})


if __name__ == '__main__':
    app.run(debug=True)
