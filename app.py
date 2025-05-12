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

os.makedirs(STUDENT_DATA_DIR, exist_ok=True)

# Load face encodings
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        known_face_encodings = pickle.load(f)
else:
    known_face_encodings = {}

# Initialize student files
def initialize_student_info():
    if not os.path.exists(STUDENT_INFO_FILE):
        df = pd.DataFrame(columns=['Student ID', 'Name'])
        df.to_csv(STUDENT_INFO_FILE, index=False)

    if not os.path.exists(STUDENT_EXCEL_FILE):
        df = pd.DataFrame(columns=['Student ID', 'Name'])
        df.to_excel(STUDENT_EXCEL_FILE, index=False)

initialize_student_info()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/student_data/<student_folder>/<filename>')
def student_image(student_folder, filename):
    return send_from_directory(os.path.join(STUDENT_DATA_DIR, student_folder), filename)

@app.route('/create_new', methods=['GET', 'POST'])
def create_new():
    if request.method == 'POST':
        name = request.form['name']
        student_id = request.form['student_id']
        save_student_info(student_id, name)
        return render_template('capture_photos.html', name=name, student_id=student_id)
    return render_template('create_new.html')

def save_student_info(student_id, name):
    df_csv = pd.read_csv(STUDENT_INFO_FILE) if os.path.exists(STUDENT_INFO_FILE) else pd.DataFrame(columns=['Student ID', 'Name'])
    if str(student_id) in df_csv['Student ID'].astype(str).values:
        print(f" Student ID {student_id} already exists. Skipping.")
        return
    df_csv = pd.concat([df_csv, pd.DataFrame([[student_id, name]], columns=['Student ID', 'Name'])], ignore_index=True)
    df_csv.to_csv(STUDENT_INFO_FILE, index=False)

    df_excel = pd.read_excel(STUDENT_EXCEL_FILE) if os.path.exists(STUDENT_EXCEL_FILE) else pd.DataFrame(columns=['Student ID', 'Name'])
    df_excel = pd.concat([df_excel, pd.DataFrame([[student_id, name]], columns=['Student ID', 'Name'])], ignore_index=True)
    df_excel.to_excel(STUDENT_EXCEL_FILE, index=False)

@app.route('/save_image', methods=['POST'])
def save_image():
    if 'image' not in request.files or 'student_id' not in request.form or 'name' not in request.form:
        return jsonify({"success": False, "message": "Missing image, student ID, or name"}), 400

    student_id = request.form['student_id']
    name = request.form['name']
    image_file = request.files['image']

    student_folder_name = f"{name}_{student_id}"
    student_folder = os.path.join(STUDENT_DATA_DIR, student_folder_name)
    os.makedirs(student_folder, exist_ok=True)

    image_filename = f"{student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    image_path = os.path.join(student_folder, image_filename)
    image_file.save(image_path)

    update_face_encodings(student_folder_name, image_path)

    return jsonify({"success": True, "message": "Image saved successfully."}), 200

def update_face_encodings(folder_name, image_path):
    global known_face_encodings
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if encodings:
        known_face_encodings[folder_name] = encodings[0]
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(known_face_encodings, f)

@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    if request.method == 'POST':
        try:
            data = request.get_json()
            image_data = data['image']
            header, encoded = image_data.split(',', 1)
            image = base64.b64decode(encoded)
            image = face_recognition.load_image_file(BytesIO(image))

            uploaded_face_encoding = face_recognition.face_encodings(image)
            if not uploaded_face_encoding:
                return jsonify({"success": False, "message": "No face found"}), 400

            uploaded_face_encoding = uploaded_face_encoding[0]
            known_encodings_list = np.array(list(known_face_encodings.values()))
            known_ids = list(known_face_encodings.keys())

            if known_encodings_list.size == 0:
                return jsonify({"success": False, "message": "No registered students"}), 400

            face_distances = face_recognition.face_distance(known_encodings_list, uploaded_face_encoding)
            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index] < 0.5:
                folder_name = known_ids[best_match_index]
                mark_attendance(folder_name)
                return jsonify({"success": True, "message": f"Attendance marked for {folder_name}"}), 200
            else:
                return jsonify({"success": False, "message": "Face not recognized"}), 400

        except Exception as e:
            return jsonify({"success": False, "message": str(e)}), 500

    return render_template('attendance.html')

# ✅ Attendance stored using Folder Name
def mark_attendance(folder_name):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_excel(ATTENDANCE_FILE)
    else:
        df = pd.DataFrame(columns=['Folder Name', 'Timestamp'])

    today = datetime.now().strftime('%Y-%m-%d')
    already_marked = (df['Folder Name'] == folder_name) & (df['Timestamp'].astype(str).str.startswith(today))

    if not already_marked.any():
        df = pd.concat([df, pd.DataFrame([[folder_name, timestamp]], columns=['Folder Name', 'Timestamp'])], ignore_index=True)
        df.to_excel(ATTENDANCE_FILE, index=False)
        print(f"✅ Attendance marked: {folder_name} at {timestamp}")
    else:
        print(f"⚠️ Attendance already marked today for: {folder_name}")

if __name__ == '__main__':
    app.run(debug=True)
