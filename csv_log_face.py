import os
import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Function to load images from a folder path and extract facial encodings
def load_images_from_folder(folder_path):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(os.path.splitext(filename)[0])

    return known_face_encodings, known_face_names

# Function to append data to CSV file
def append_to_csv(name, entry_time):
    with open('face_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, entry_time])

# Path to the folder containing images of known faces
folder_path = "photos/"

# Load images from the folder and extract facial encodings
known_face_encodings, known_face_names = load_images_from_folder(folder_path)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Set to keep track of logged names
logged_names = set()

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Threshold distance to consider a face as unknown
threshold_distance = 0.4  # You can adjust this value as needed

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Check if any of the distances are below the threshold
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index] and face_distances[best_match_index] < threshold_distance:
                name = known_face_names[best_match_index]
                # Log only if not already logged
                if name not in logged_names:
                    entry_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    append_to_csv(name, entry_time)
                    logged_names.add(name)

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

