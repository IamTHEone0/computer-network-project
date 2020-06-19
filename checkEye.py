from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import sys

def _help():
    print("Usage:")
    print("     python blink_detect.py")
    print("     python blink_detect.py <path of a video>")
    print("For example:")
    print("     python blink_detect.py video/lee.mp4")
    print("If the path of a video is not provided, the camera will be used as the input.Press q to quit.")

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def blink_detection(vs, file_stream):
    # define three constants, one for the eye aspect ratio to indicate
    # blink and then the other constants for the min/max number of consecutive
    # frames the eye must be below the threshold
    EAR_THRESH = 0.2
    EAR_CONSEC_FRAMES_MIN = 1
    EAR_CONSEC_FRAMES_MAX = 2
    # initialize the frame counters and the total number of blinks
    blink_counter = [0, 0]  # left eye and right eye
    blink_total = [0, 0]  # left eye and right eye
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    print("[INFO] starting video stream thread...")
    while True:
        # if this is a file video stream, then we need to check if
        # there any more frames left in the buffer to process
        if file_stream and not vs.more():
            break
        frame = vs.read()
        if frame is not None:
            frame = imutils.resize(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            if len(rects) == 1:
                rect = rects[0]
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                left_eye = shape[lStart:lEnd]
                right_eye = shape[rStart:rEnd]
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                # compute the convex hull for the left and right eye, then
                # visualize each of the eyes
                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
                # check to see if the eye aspect ratio is below the blink
                # threshold, and if so, increment the blink frame counter
                if left_ear < EAR_THRESH:
                    blink_counter[0] += 1

                # otherwise, the eye aspect ratio is not below the blink
                # threshold
                else:
                    # if the eyes were closed for a sufficient number of
                    # then increment the total number of blinks
                    if EAR_CONSEC_FRAMES_MIN <= blink_counter[0] and blink_counter[0] <= EAR_CONSEC_FRAMES_MAX:
                        blink_total[0] += 1
                        print(blink_total)
                    blink_counter[0] = 0
                # draw the total number of blinks on the frame along with
                # the computed eye aspect ratio for the frame
                cv2.putText(frame, "LBlinks: {}".format(blink_total[0]), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "LEAR: {:.2f}".format(left_ear), (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # check to see if the eye aspect ratio is below the blink
                # threshold, and if so, increment the blink frame counter
                if right_ear < EAR_THRESH:
                    blink_counter[1] += 1

                # otherwise, the eye aspect ratio is not below the blink
                # threshold
                else:
                    # if the eyes were closed for a sufficient number of
                    # then increment the total number of blinks
                    if EAR_CONSEC_FRAMES_MIN <= blink_counter[1] and blink_counter[1] <= EAR_CONSEC_FRAMES_MAX:
                        blink_total[1] += 1
                        print(blink_total)
                    blink_counter[1] = 0
                # draw the total number of blinks on the frame along with
                # the computed eye aspect ratio for the frame
                cv2.putText(frame, "RBlinks: {}".format(blink_total[1]), (200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "REAR: {:.2f}".format(right_ear), (200, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif len(rects) == 0:
                cv2.putText(frame, "No face!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "More than one face!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
            cv2.imshow("Frame", frame)
            # if the `q` key was pressed, break from the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
    vs.stop()

# if len(sys.argv) > 2 or "-h" in sys.argv or "--help" in sys.argv:
#     _help()
# elif len(sys.argv) == 2:
#     vs = FileVideoStream(sys.argv[1]).start()
#     file_stream = True
#     blink_detection(vs, file_stream)
# else:
#     vs = VideoStream(src=0).start()
#     file_stream = False
#     blink_detection(vs, file_stream)
vs = FileVideoStream('data_src.mp4').start()
file_stream = True
blink_detection(vs, file_stream)