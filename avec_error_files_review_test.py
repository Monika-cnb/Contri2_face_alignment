# Code to analyse the files that encountered an error during face-alignment code implementation
import os
import pandas as pd
import dlib
import cv2

test_error_file_list = ["246_2_cut_combined.mp4", "345_3_cut_combined.mp4", "242_1_cut_combined.mp4",
                        "218_3_cut_combined.mp4", "357_2_cut_combined.mp4"]

base_path = "./../../Desktop/Contri_2_updated/0_face_alignment/"
print(os.listdir(base_path))
input_file_path = base_path + "0_aligned_dataset/0_avec/error_files/"
output_file_path = base_path + "0_aligned_dataset/0_avec/error_files_corrected/"

# Reviewing the train set files
for each_video in test_error_file_list:
    print(each_video)
    # file_path = input_file_path + each_file
    # Load the shape predictor model
    shape_predictor_path = base_path + "0_code/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(shape_predictor_path)

    # Load the face detector
    detector = dlib.get_frontal_face_detector()

    # Load the video
    video_path = input_file_path + each_video[:-4] + ".avi"
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # # Variable to count frame - to start after the interview location
    # frame_count = -1
    # frames_to_read = fps * 10000

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # reading the output file
    output_path = output_file_path + each_video[:-4] + ".avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Use MP4 codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    # out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))
    left, right, top, bottom = 0, 0, 0, 0
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
            # print(frame_count)
            print("All values: left, right, top, bottom", left, right, top, bottom)
            print(frame_width, frame_height)
            break
        break

    old_left, old_top, old_right, old_bottom = left - 100, top - 100, right + 100, bottom + 100
    print("All values: old_left, old_right, old_top, old_bottom", old_left, old_right, old_top, old_bottom)
    # if value is negative, make it zero
    if old_left < 0:
        old_left = 0
    if old_top < 0:
        old_top = 0
    # if value is greater than frame size, make it equal to frame size
    if old_right > frame_width:
        old_right = frame_width
    if old_bottom > frame_height:
        old_bottom = frame_height

    while True:
        # frame_count += 1
        # current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # print(f"Current Frame: {current_frame}")
        # print(frame_count)
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

        # Write the cropped face region to the ImageWriter
        #

    cap.release()
    cv2.destroyAllWindows()
