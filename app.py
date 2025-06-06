import streamlit as st
import cv2
import face_recognition
import os
import time
from datetime import datetime
from PIL import Image
import uuid
import base64

# === CONFIGURATION ===
KNOWN_FACES_DIR = "images"
VIDEO_SOURCE = "SIS_cv2.mp4"
ALERT_SOUND_PATH = "alert.wav"
SNAPSHOT_DIR = "snapshots"
ALERT_COOLDOWN = 5

# Ensure snapshot directory exists
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# === Load known faces ===
@st.cache_data
def load_known_faces():
    encodings = []
    names = []
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(path)
            face_encs = face_recognition.face_encodings(image)
            if face_encs:
                encodings.append(face_encs[0])
                names.append(os.path.splitext(filename)[0])
    return encodings, names

# === Embed audio in HTML ===
def get_audio_html(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        return f"""
            <audio autoplay>
                <source src="data:audio/wav;base64,{b64}" type="audio/wav">
            </audio>
        """

# === Streamlit Styling ===
st.set_page_config(page_title="SIS", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #f9f9f9;}
    h1, h2 {color: #1a1a1a;}
    .snapshot {border: 2px solid red; border-radius: 10px; padding: 5px;}
    </style>
""", unsafe_allow_html=True)

# === Sidebar Navigation ===
page = st.sidebar.selectbox("Navigation", ["Home", "Employee Database"])

# === Load known faces once ===
known_face_encodings, known_face_names = load_known_faces()

# === Home Page ===
if page == "Home":
    st.title("SIS - Surveillance Information System")

    # === Section 1: YouTube Reference Stream ===
    st.subheader("🔴 Before SIS")
    st.video("https://www.youtube.com/watch?v=j3E27LLmDXc")

    # === Section 2: Real-time CV2 Detection ===
    st.subheader("📸 After SIS")

    frame_placeholder = st.empty()
    snapshot_placeholder = st.empty()

    video_capture = cv2.VideoCapture(VIDEO_SOURCE)
    last_alert_time = 0
    snapshot_taken = False

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_labels = []
        intruder_detected = False

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if True in matches:
                match_index = matches.index(True)
                name = known_face_names[match_index]
                label = f"Member: {name}"
            else:
                label = "Non-Member"
                intruder_detected = True
            face_labels.append(label)

        # Draw bounding boxes
        for (top, right, bottom, left), label in zip(face_locations, face_labels):
            top *= 4; right *= 4; bottom *= 4; left *= 4
            color = (0, 0, 255) if label == "Non-Member" else (0, 255, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, top - 20), (right, top), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, top - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        # === Take and display snapshot only once during cooldown ===
        current_time = time.time()
        if intruder_detected and (current_time - last_alert_time > ALERT_COOLDOWN):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{SNAPSHOT_DIR}/intruder_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            snapshot_placeholder.subheader("Intruder Snapshot")
            snapshot_img = Image.open(filename)
            snapshot_placeholder.image(snapshot_img, caption="🚨 Intruder Detected!", use_container_width=True)
            st.markdown(get_audio_html(ALERT_SOUND_PATH), unsafe_allow_html=True)
            last_alert_time = current_time
            snapshot_taken = True

        # Show frame in container
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    video_capture.release()

# === Employee Database Page ===
elif page == "Employee Database":
    st.title("👥 Registered Employees")
    cols = st.columns(3)

    for i, filename in enumerate(os.listdir(KNOWN_FACES_DIR)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            name = os.path.splitext(filename)[0]
            emp_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, name))[:8].upper()
            with cols[i % 3]:
                st.image(path, use_container_width=True)
                st.markdown(f"**Name:** {name}")
                st.markdown(f"**Employee Code:** `{emp_id}`")
