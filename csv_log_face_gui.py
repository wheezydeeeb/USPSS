import os
import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import shutil
import tkinter as tk
from tkinter import ttk

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

# Function to move CSV log file to a separate folder
def move_csv_to_folder():
    source_file = 'face_log.csv'
    destination_folder = 'csv_logs'

    # Create a new CSV file if it doesn't exist
    if not os.path.exists(source_file):
        with open(source_file, mode='w', newline=''):
            pass
    
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Generate filename using start and end time
    start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    end_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    new_filename = f'{start_time}_{end_time}.csv'
    
    # Move the CSV file to the destination folder
    shutil.move(source_file, os.path.join(destination_folder, new_filename))

# Function to display the file tree of a directory using Tkinter
def display_file_tree(directory):
    root = tk.Tk()
    root.title("File Tree")

    tree = ttk.Treeview(root)
    tree.pack(fill='both', expand=True)

    def populate_tree(parent, path):
        for p in os.listdir(path):
            full_path = os.path.join(path, p)
            if os.path.isdir(full_path):
                oid = tree.insert(parent, 'end', text=p, open=False)
                populate_tree(oid, full_path)
            else:
                tree.insert(parent, 'end', text=p)

    populate_tree('', directory)
    root.mainloop()

# Path to the folder containing images of known faces
folder_path = "photos/"

# Load images from the folder and extract facial encodings
known_face_encodings, known_face_names = load_images_from_folder(folder_path)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

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
            elif name != "Unknown":  # Only log if not an unknown face and not already logged
                entry_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                if name not in face_names:
                    append_to_csv(name, entry_time)

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
        # Move CSV log file to a separate folder
        move_csv_to_folder()
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

# Display file tree of the directory containing CSV log files
display_file_tree('csv_logs')

