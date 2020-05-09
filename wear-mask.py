import dlib
from PIL import Image, ImageDraw, ImageFont
import argparse

import cv2

from imutils.video import VideoStream
from imutils import face_utils, translate, rotate, resize

import numpy as np


print("Press 'q' to quit...")
print("Press 'm' to wear a mask...")

vs = VideoStream().start()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

max_width = 500
frame = vs.read()
frame = resize(frame, width=max_width)

fps = vs.stream.get(cv2.CAP_PROP_FPS) # need this for animating proper duration

animation_length = fps * 3
current_animation = 0
glasses_on = fps * 1

text = Image.open('text2.png')
mask = Image.open("mask.png")

wearing = False


while True:
    frame = vs.read()

    frame = resize(frame, width=max_width)

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = []

    rects = detector(img_gray, 0)
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    for rect in rects:
        face = {}
        mask_width = rect.right() - rect.left()

        # find current face orientation
        shape = predictor(img_gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the outlines of the face
        jawline = shape[:17]
        nose_center = shape[30]

	    # compute the angle the face is rotated
        dY = jawline[0,1] - jawline[16,1] 
        dX = jawline[0,0] - jawline[16,0]
        angle = np.rad2deg(np.arctan2(dY, dX)) 

        current_mask = mask.resize((int(1.25*mask_width), int(1.25 * mask_width * mask.size[1] / mask.size[0])), resample=Image.LANCZOS)
        current_mask = current_mask.rotate(angle, expand=True)
        current_mask = current_mask.transpose(Image.FLIP_TOP_BOTTOM)

        face['mask_image'] = current_mask
        left_x = jawline[0,0] - mask_width // 8
        left_y = jawline[0,1] - mask_width // 5
        face['final_pos'] = (left_x, left_y)

        if wearing:
            if current_animation < glasses_on:
                current_y = int(current_animation / glasses_on * left_y)
                img.paste(current_mask, (left_x, current_y), current_mask)
            else:
                img.paste(current_mask, (left_x, left_y), current_mask)
                img.paste(text, (0, 20), text)

    if wearing:
        current_animation += 1

        # to save pngs for creating gifs, videos
        img.save("images/%05d.png" % current_animation)

        if current_animation > animation_length:
            wearing = False
            current_animation = 0
        else:
            frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    cv2.imshow("mask generator", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("Exiting...")
        break

    if key == ord("m"):
        wearing = not wearing

cv2.destroyAllWindows()
vs.stop()