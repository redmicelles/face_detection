import cv2
import streamlit as st
import os

# Create a folder for saving images
output_dir = "detected_faces"
os.makedirs(output_dir, exist_ok=True)

# Set page title and description
st.title("FACIAL DETECTION")
st.markdown("<h6 style='text-align: right; color: #FAF8F1'>By Makanju</h6>", unsafe_allow_html=True)

# Load and display image
st.sidebar.image("pngwing.com (20).png")
# img = cv2.imread('pngwing.com (20).png')
# st.image(img, caption="Facial attribute detector", width=500)

# User registration
username = st.text_input('Please register your name', key='username')
if st.button('Submit name'):
    st.success(f"Welcome {username}.")

# Face detection section
st.write("Welcome to Face Detection using Viola-Jones Algorithm!")
st.write("Instructions:")
st.write("1. Position yourself in front of the camera.")
st.write("2. The app will detect your face and draw rectangles around it.")
st.write("3. Adjust the parameters below to customize the detection.")
st.write("4. Press 'q' to exit the app.")

# Parameters for face detection
min_neighbors = st.slider("Adjust minNeighbors", 1, 10, 5)
scale_factor = st.slider("Adjust scaleFactor", 1.1, 2.0, 1.3)

# Start face detection on camera stream
if st.button('Start Camera'):
    cascade_path = 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # or cv2.CAP_V4L2

    exit_app = False  # Flag to track the exit status

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 255, 0), 2)

            # Save the image with detected face
            output_path = os.path.join(output_dir, f"detected_face_{username}.jpg")
            cv2.imwrite(output_path, frame)

        cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)

        if cv2.waitKey(1) == ord('q') or exit_app:
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Exit button and functionality
if st.button('Exit'):
    exit_app = True
