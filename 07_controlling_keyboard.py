import cv2
import numpy as np
import dlib
from math import hypot
import virtual_keyboard_2
import pyglet
import time

#SOUND
sound = pyglet.media.load("sound.wav", streaming=False)
left_sound = pyglet.media.load("left.wav", streaming=False)
right_sound = pyglet.media.load("right.wav", streaming=False)

cap = cv2.VideoCapture(0)
board = np.zeros((500, 500), np.uint8)
board[:] = 255

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

font = cv2.FONT_HERSHEY_SIMPLEX

keys_set = {0: "Q", 1: "W", 2: "E", 3: "R", 4: "T",
              5: "A", 6: "S", 7: "D", 8: "F", 9: "G",
              10: "Z", 11: "X", 12: "C", 13: "V", 14: "B"}


def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    ratio = hor_line_lenght / ver_line_lenght
    return ratio

def get_gaze_ratio(eye_points, facial_landmarks):
    # gaze detection
    eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)

    # mask
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [eye_region], True, 255, 2)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    return  gaze_ratio


#Counters
frames = 0
letter_index = 0
blinking_frames = 0
text = ""
keyboard_selected = "left"
last_keyboard_selected = "left"
while True:
    _, frame = cap.read()
    keyboard[:] = (0, 0, 0)
    frame = cv2.resize(frame,None, fx=0.5, fy=0.5)
    frames += 1
    new_frame = np.zeros((500, 500, 3), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    active_letter = keys_set[letter_index]

    faces = detector(gray)
    for face in faces:
        # x, y = face.left(), face.top()
        # x1, y1 = face.right(), face.bottom()
        # cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        left_point = (landmarks.part(36).x, landmarks.part(36).y)
        right_point = (landmarks.part(39).x, landmarks.part(39).y)

        center_top = midpoint(landmarks.part(37), landmarks.part(38))
        center_bottom = midpoint(landmarks.part(41), landmarks.part(40))
        hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
        ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

        # to get length of horizontal line:
        #hor_line_length = hypot(((left_point[0] - right_point[0])),(left_point[1] - right_point[1]))

        # to get length of vertical line
        #ver_line_length = hypot(((center_top[0] - center_bottom[0])), (center_top[1] - center_bottom[1]))

        #print(hor_line_length/ver_line_length)

        landmarks = predictor(gray, face)
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
        if blinking_ratio > 5.7:
            cv2.putText(frame, "BLINKING", (50, 150), font, 7, (255, 0, 0),thickness=3)
            blinking_frames += 1
            frames -= 1

            if blinking_frames == 5:
                text += active_letter
                sound.play()
                time.sleep(1)
        else:
            blinking_frames = 0

        # GAZE DETECTON
        gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        gaze_ratio = (gaze_ratio_left_eye + gaze_ratio_right_eye) / 2

        # cv2.putText(frame, str(gaze_ratio), (50,100), font, 2, (0, 0, 255), 3)
        new_frame = np.zeros((500, 500, 3), np.uint8)

        if gaze_ratio <= 0.9:
            keyboard_selected = "right"
            if keyboard_selected != last_keyboard_selected:
                right_sound.play()
                time.sleep(1)
                last_keyboard_selected = keyboard_selected
        else:
            keyboard_selected = "left"
            if keyboard_selected != last_keyboard_selected:
                left_sound.play()
                time.sleep(1)
                last_keyboard_selected = keyboard_selected

    # Letters
    if frames == 10:
        letter_index += 1
        frames = 0
    if letter_index == 15:
        letter_index = 0

    keyboard = virtual_keyboard_2.call(keys_set,letter_index)
    #keyboard[:] = (0, 0, 0)

    cv2.putText(board, text, (10, 100), font, 4, 0, 3)

    cv2.imshow("Frame", frame)
    #cv2.imshow("New frame", new_frame)
    cv2.imshow("keyboard", keyboard)
    cv2.imshow("BOARD", board)
    if cv2.waitKey(33) == ord('a'):
        break
cap.release()
cv2.destroyAllWindows()