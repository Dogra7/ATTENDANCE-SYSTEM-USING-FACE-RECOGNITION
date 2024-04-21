import os
import cv2
import numpy as np
import json
import psycopg2
from psycopg2 import Error
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, Response
from datetime import datetime
import face_recognition

FACE_DATA_DIR = 'face_data'

app = Flask(__name__)
app.secret_key = 'deep7'
app.config['UPLOAD_FOLDER'] = 'faces'

users = {
    'deep': 'deep123',
    'drash': 'drash123',
    'sneha': 'sneha123',
    'ruchita': 'ruchita123'
}

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
        print("Recognized face name:", recognized_face_name)  # Add this line for debugging
        # Establish a connection to the database
        connection = connect_to_database()
        if connection:
            cursor = connection.cursor()

            # Check if recognized face name is not "Unknown"
            if recognized_face_name != "Unknown":
                # Prepare SQL statement to insert attendance record
                insert_query = """INSERT INTO attendance_record (s_name) VALUES (%s)"""
                record_to_insert = (recognized_face_name,)

                # Execute the SQL statement
                cursor.execute(insert_query, record_to_insert)

                # Commit the transaction
                connection.commit()
                print("Attendance record inserted successfully:", recognized_face_name)
            else:
                print("Skipping 'Unknown' face for attendance record.")

    except (Exception, Error) as error:
        print("Error while inserting attendance record:", error)

    finally:
        # Close the database connection
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed.")


def fetch_attendance_records():
    """Fetches attendance records from the database."""
    try:
        # Establish a connection to the database
        connection = connect_to_database()
        if connection:
            cursor = connection.cursor()

            # Fetch all attendance records from the 'attendance' table
            cursor.execute("SELECT * FROM attendance_record")

            # Retrieve all rows from the result set
            attendance_records = cursor.fetchall()

            print("Fetched attendance records successfully:", attendance_records)  # Add this print statement

            return attendance_records

    except (Exception, Error) as error:
        print("Error while fetching attendance records:", error)

    finally:
        # Close the database connection
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed.")


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

            # print("Recognized face name:", name)

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
    success = store_attendance_in_database(recognized_face_name)
    if success:
        return jsonify({'success': True})
    else:
        return jsonify({'success': False})
    
def extract_features(face_roi):
    return np.random.rand(128)


@app.route('/mark_attendance', methods=['GET', 'POST'])
def mark_attendance():
    return render_template('mark_attendance_webcam.html')


@app.route('/attendance_report', methods=['GET','POST'])
def attendance_report():
    attendance_records = fetch_attendance_records()
    print("Fetched attendance records successfully:", attendance_records)  # Debug statement
    return render_template('attendance_report.html', attendance_records=attendance_records)

@app.route('/login', methods=['GET', 'POST'])
def login():
    return render_template('login.html')

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in users and users[username] == password:
            session['username'] = username
            return render_template('admin.html')
        else:
            return render_template('error.html')
    return render_template('admin.html')

@app.route('/add_face', methods=['POST'])
def add_face():
    file = request.files['image']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    name = os.path.splitext(file.filename)[0]
    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) != 1:
        return "Error: Please upload an image with exactly one face"
    x, y, w, h = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
    face_file_name = f'{name}.jpg'
    face_file_path = os.path.join(app.config['UPLOAD_FOLDER'], face_file_name)
    cv2.imwrite(face_file_path, face_roi)
    face_features = extract_features(face_roi)
    face_data = {
        'name': name,
        'timestamp': timestamp,
        'file_path': face_file_name,
        'features': face_features.tolist()
    }
    if not os.path.exists(FACE_DATA_DIR):
        os.makedirs(FACE_DATA_DIR)
    face_data_file_path = os.path.join(FACE_DATA_DIR, f'{name}_{timestamp}.json')
    with open(face_data_file_path, 'w') as json_file:
        json.dump(face_data, json_file)
    return redirect(url_for('admin'))

@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

if __name__ == '__main__':    
    app.run(debug=True)