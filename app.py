from flask import Flask, render_template, request, jsonify, send_from_directory
import base64
import os
import pandas as pd
import face_recognition
import pickle
import numpy as np
from datetime import datetime 
from io import BytesIO

app = Flask(__name__)

# Directories & Files
STUDENT_DATA_DIR = "student_data"
ATTENDANCE_FILE = "attendance.xlsx"
STUDENT_INFO_FILE = "student_info.csv"
STUDENT_EXCEL_FILE = "students.xlsx"
ENCODINGS_FILE = "face_encodings.pkl"

# Ensure necessary directories exist
os.makedirs(STUDENT_DATA_DIR, exist_ok=True)

# Load student encodings if available
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        known_face_encodings = pickle.load(f)
else:
    known_face_encodings = {}

# Ensure student_info.csv and students.xlsx exist
def initialize_student_info():
    if not os.path.exists(STUDENT_INFO_FILE):
        df = pd.DataFrame(columns=['Student ID', 'Name'])
        df.to_csv(STUDENT_INFO_FILE, index=False)

    if not os.path.exists(STUDENT_EXCEL_FILE):
        df = pd.DataFrame(columns=['Student ID', 'Name'])
        df.to_excel(STUDENT_EXCEL_FILE, index=False)

initialize_student_info()

# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Serve images from student_data directory
@app.route('/student_data/<student_id>/<filename>')
def student_image(student_id, filename):
    return send_from_directory(os.path.join(STUDENT_DATA_DIR, student_id), filename)

# Register a new student
@app.route('/create_new', methods=['GET', 'POST'])
def create_new():
    if request.method == 'POST':
        name = request.form['name']
       
        student_id = request.form['student_id']
        
       
        # Save student details
        save_student_info(student_id, name)

        return render_template('capture_photos.html', name=name, student_id=student_id)
    
    return render_template('create_new.html')

# Save student info in CSV & Excel
def save_student_info(student_id, name):
    # Load CSV
    df_csv = pd.read_csv(STUDENT_INFO_FILE) if os.path.exists(STUDENT_INFO_FILE) else pd.DataFrame(columns=['Student ID', 'Name'])

    # Prevent duplicate entries
    if str(student_id) in df_csv['Student ID'].astype(str).values:
        print(f" Student ID {student_id} already exists. Skipping.")
        return  

    # Append new student info
    df_csv = pd.concat([df_csv, pd.DataFrame([[student_id, name]], columns=['Student ID', 'Name'])], ignore_index=True)
    df_csv.to_csv(STUDENT_INFO_FILE, index=False)

    # Save in Excel
    df_excel = pd.read_excel(STUDENT_EXCEL_FILE) if os.path.exists(STUDENT_EXCEL_FILE) else pd.DataFrame(columns=['Student ID', 'Name'])
    df_excel = pd.concat([df_excel, pd.DataFrame([[student_id, name]], columns=['Student ID', 'Name'])], ignore_index=True)
    df_excel.to_excel(STUDENT_EXCEL_FILE, index=False)

    print(f" Student {name} (ID: {student_id}) added successfully!")

# Save student images & update encodings
@app.route('/save_image', methods=['POST'])
def save_image():
    if 'image' not in request.files or 'student_id' not in request.form:
        return jsonify({"success": False, "message": "Missing image or student ID"}), 400

    student_id = request.form['student_id']
    name=request.form['name']
    image_file = request.files['image']

    # Create student folder
    student_folder = os.path.join(STUDENT_DATA_DIR, student_id)
    os.makedirs(student_folder, exist_ok=True)

    # Save image
    image_filename = f"{student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    image_path = os.path.join(student_folder, image_filename)
    image_file.save(image_path)

    # Update face encodings
    update_face_encodings(student_id, image_path,name)

    return jsonify({"success": True, "message": "Image saved successfully."}), 200

# Update face encodings
def update_face_encodings(student_id, image_path,name):
    global known_face_encodings

    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if encodings:
        known_face_encodings[student_id] = encodings[0]

        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(known_face_encodings, f)
        print(f"✅ Encodings updated for student {student_id} {name}")

# Mark attendance
@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    if request.method == 'POST':
        try:
            data = request.get_json()
            image_data = data['image']

            # Decode and load image
            header, encoded = image_data.split(',', 1)
            image = base64.b64decode(encoded)
            image = face_recognition.load_image_file(BytesIO(image))

            # Get encodings
            uploaded_face_encoding = face_recognition.face_encodings(image)
            if not uploaded_face_encoding:
                return jsonify({"success": False, "message": "No face found"}), 400

            uploaded_face_encoding = uploaded_face_encoding[0]

            # Fast face matching using NumPy
            known_encodings_list = np.array(list(known_face_encodings.values()))
            known_ids = list(known_face_encodings.keys())

            if known_encodings_list.size == 0:
                return jsonify({"success": False, "message": "No registered students"}), 400

            face_distances = face_recognition.face_distance(known_encodings_list, uploaded_face_encoding)
            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index] < 0.5:
                best_match_id = known_ids[best_match_index]
                best_match_name = get_student_name(best_match_id)
                mark_attendance(best_match_id, best_match_name)
                return jsonify({"success": True, "message": "Attendance marked successfully"}), 200
            else:
                return jsonify({"success": False, "message": "Face not recognized"}), 400

        except Exception as e:
            return jsonify({"success": False, "message": str(e)}), 500

    return render_template('attendance.html')

# Mark attendance in Excel
def mark_attendance(student_id, name):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_excel(ATTENDANCE_FILE)
    else:
        df = pd.DataFrame(columns=['Student ID', 'Name', 'Timestamp'])

    # Check if attendance for today is already marked
    today = datetime.now().strftime('%Y-%m-%d')
    if not ((df['Student ID'] == student_id) & (df['Timestamp'].astype(str).str.startswith(today))).any():
        df = pd.concat([df, pd.DataFrame([[student_id, name, timestamp]], columns=['Student ID', 'Name', 'Timestamp'])], ignore_index=True)
        df.to_excel(ATTENDANCE_FILE, index=False)
        print(f" Attendance marked: {name} (ID: {student_id}) at {timestamp}")
    else:
        print(f" Attendance already marked today for: {name} (ID: {student_id})")

# Get student name from CSV
def get_student_name(student_id):
    df = pd.read_csv(STUDENT_INFO_FILE)
    student_info = df[df['Student ID'].astype(str) == str(student_id)]
    return student_info.iloc[0]['Name'] if not student_info.empty else "Unknown"

if __name__ == '__main__':
    app.run(debug=True)    