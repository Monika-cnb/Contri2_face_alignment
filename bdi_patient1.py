# just reviewing code to complete for face alignment - for patient dataset

import dlib
import cv2
import imageio
import os
import subprocess
import pandas as pd
import numpy as np

# define path for the video files
initial_data_path = "./../../timo/datasets/AV/BDI/"

# Defining folders which we use for extracting openface values
training_data_path = initial_data_path + "Compressed/Patients/"

# path for saving the aligned files
output_dir = "./../../Desktop/Contri_2_updated/0_face_alignment/"
output_dir_path_train = output_dir + "0_aligned_dataset/0_bdi/0_patient/"
# code to read the patient dataframe
patient_df = pd.read_csv(output_dir + "0_aligned_dataset/0_bdi/Patient_file_details.csv")
print(patient_df)
all_train_files = ['Pa_PF1-4354_NOMS066.avi', 'Pa_PF1-4439_NOMS070.avi', 'Pa_PF1-4631_NOMS081.avi', 'Pa_StCo.avi', 'Pa_IMC-2911.avi', 'Pa_MBI-3713_noms021.avi', 'Pa_MH0-4536_NOMS071.avi', 'Pa_GP10_NOMS054.avi', 'Pa_HS1-4414_NOMS073.avi', 'Pa_MH-3089.avi', 'Pa_MHO_3377.avi', 'Pa_MHO_3659_NOMS014.avi', 'Pa_PF1_3443.avi', 'Pa_PF1-4653_NOMS078.avi', 'Pa_HS1-4181_NOMS053.avi', 'Pa_HS1-4682_NOMS079.avi', 'Pa_MB1-3606_NOMS0015.avi', 'Pa_PF1-4623_NOMS075.avi', 'Pa_SF1-4079_NOMS049.avi', 'Pa_MHO-3686_noms019.avi', 'Pa_SF1-4016_NOMS043.avi', 'GP4_noms013.avi', 'Pa_HSI-3732_NOMS030.avi']

error_files = []
video_num = 0
# code to read each row of the above dataframe
for val in range(len(patient_df)):
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
    frame_num = total_seconds * fps
    print(fps)
    print(frame_num)

    # Calculate the number of frames up to the 5-minute mark (300 seconds) - just above 5 minutes mark
    frames_to_read = fps * 10000
    frame_count = 0
    stop
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = output_dir_path_train + each_video[:-4] + ".avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Use MP4 codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    # out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))
    while cap.isOpened():
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
            # print("All values: left, right, top, bottom", left, right, top, bottom)
            break
        break

    old_left, old_top, old_right, old_bottom = left - 100, top - 100, right + 100, bottom + 100
    while frame_count < frames_to_read:
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
