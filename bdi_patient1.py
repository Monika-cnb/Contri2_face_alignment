# just reviewing code to complete for face alignment - for patient dataset

import dlib
import cv2
# import imageio
# import os
# import subprocess
import pandas as pd
# import numpy as np

# define path for the video files
initial_data_path = "./../../timo/datasets/AV/BDI/"

# Defining folders which we use for extracting openface values
training_data_path = initial_data_path + "Compressed/Patients/"

# path for saving the aligned files
output_dir = "./../../Desktop/Contri_2_updated/0_face_alignment/"
output_dir_path_train = output_dir + "0_aligned_dataset/0_bdi/0_patient1/"
# code to read the patient dataframe
patient_df = pd.read_csv(output_dir + "0_aligned_dataset/0_bdi/Patient_file_details.csv")
print(patient_df)

error_files = []
video_num = 0
# code to read each row of the above dataframe
for val in range(len(patient_df)):
    try:
        each_video = patient_df['File'][val] + ".avi"
        vid_time_dur = patient_df['interview'][val]
        minutes, seconds, ex = vid_time_dur.split(":")
        total_seconds = int(minutes) * 60 + int(seconds)

        # read each file
        print("Starting video num ", video_num)
        video_num += 1
        print(each_video)
        # Load the shape predictor model
        shape_predictor_path = output_dir + "0_code/shape_predictor_68_face_landmarks.dat"
        predictor = dlib.shape_predictor(shape_predictor_path)

        # Load the face detector
        detector = dlib.get_frontal_face_detector()

        # Load the video
        video_path = training_data_path + each_video
        cap = cv2.VideoCapture(video_path)

        # Get the frames per second (fps) and frame size of the input video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        # interview location start for each video
        start_frame = total_seconds * fps
        print(fps)
        print(start_frame)

        # Variable to count frame - to start after the interview location
        frame_count = -1
        frames_to_read = fps * 10000

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # reading the output file
        output_path = output_dir_path_train + each_video[:-4] + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Use MP4 codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        # out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))
        left, right, top, bottom = 0, 0, 0, 0
        while cap.isOpened():
            frame_count += 1
            # if we haven't reached the actual interview location, skip the frames
            if frame_count < start_frame:
                print(frame_count)
                continue
            print("Actually reading the frame")

            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale frame
            faces = detector(gray)
            for face in faces:
                # Predict facial landmarks
                landmarks = predictor(gray, face)

                # Get bounding box coordinates of the aligned face
                left, top, right, bottom = (
                    max(face.left(), 0),
                    max(face.top(), 0),
                    min(face.right(), frame.shape[1]),
                    min(face.bottom(), frame.shape[0]),
                )
                print(frame_count)
                print("All values: left, right, top, bottom", left, right, top, bottom)
                break
            break

        old_left, old_top, old_right, old_bottom = left - 100, top - 100, right + 100, bottom + 100
        while True:
            print(frame_count)
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale frame
            faces = detector(gray)
            for face in faces:
                # Predict facial landmarks

                landmarks = predictor(gray, face)

                # Get bounding box coordinates of the aligned face
                new_left, new_top, new_right, new_bottom = (
                    max(face.left(), 0),
                    max(face.top(), 0),
                    min(face.right(), frame.shape[1]),
                    min(face.bottom(), frame.shape[0]),
                )
                # print("All values: old_left, old_right, old_top, old_bottom", old_left, old_right, old_top, old_bottom)
                # print("All values: new_left, new_right, new_top, new_bottom", new_left, new_right, new_top, new_bottom)

                # Crop the aligned face region from the frame
                aligned_face_region = frame[old_top:old_bottom, old_left:old_right]
                aligned_face_region_resized = cv2.resize(
                    aligned_face_region, (frame_width, frame_height)
                )
                # print("aligned_face_region shape:", aligned_face_region.shape)
                # print("aligned_face_region dtype:", aligned_face_region.dtype)
                # print("actual face shape:", frame.shape)
                out.write(aligned_face_region_resized)
                ###out.write(frame)

                # Display the cropped face region
                # cv2.imshow("Cropped Face", aligned_face_region)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1
            # Write the cropped face region to the ImageWriter
            #

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Error processing this file:", e)
        error_files.append(each_video)

