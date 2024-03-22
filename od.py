""" stds-sample-code-for-object-detection.py
    
    Author: Jesus Leonardo Tamez Gloria
            Jose Miguel Gonzalez Zaragoza
    Organisation: Universidad de Monterrey
    Contact: jesusl.tamez@udem.edu
             jose.gonzalezz@udem.edu

    USAGE: 
    $ python .\test-object-detection.py --video_file football-field-cropped-video.mp4 --frame_resize_percentage 30
"""


import cv2 
import argparse
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Union, List


HSV_params = {'low_H': 0, 
              'high_H': 180,            #inicializamos los parametros para HSV, siendo estos sus valores maximos y minimos
              'low_S': 0,
              'high_S': 255,
              'low_V': 0,
              'high_V': 255
            }

window_params = {'capture_window_name':'Input video', #esta parte sirve para nombrar las ventanas donde se observa el el video original
                 'detection_window_name':'Detected object'} #y el video binarizado


def parse_cli_data()->argparse: #lectura del archivo de video
    parser = argparse.ArgumentParser(description='Tunning HSV bands for object detection')
    parser.add_argument('--video_file', 
                        type=str, 
                        default='camera', 
                        help='Video file used for the object detection process')
    parser.add_argument('--frame_resize_percentage', 
                        type=int, 
                        help='Rescale the video frames, e.g., 20 if scaled to 20%')
    args = parser.parse_args()

    return args


def initialise_camera(args:argparse)->cv2.VideoCapture:
    
    # Create a video capture object
    cap = cv2.VideoCapture(args.video_file)
    
    return cap


def rescale_frame(frame:NDArray, percentage:np.intc=20)->NDArray:
    
    # Resize current frame
    width = int(frame.shape[1] * percentage / 100)
    height = int(frame.shape[0] * percentage / 100)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame

def binary_colour(frame: np.ndarray) -> np.ndarray: #binarizamos el frame, por lo cual, identificamos los colores que se utilizan. No sin antes 

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #usar cambiar de bgr a hsv, pues este sirve para la identificacion de objetos

    # Define the range for blue and black in HSV
    lower_blue = np.array([100, 100, 20])
    upper_blue = np.array([125, 255, 255])
    lower_black = np.array([50, 0, 0])
    upper_black = np.array([180, 255, 33])


    # Generate masks for blue and black
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_black = cv2.inRange(hsv, lower_black, upper_black) #hacemos una mascara para cada color, con sus respectivos limites

    # Combine masks
    combined_mask = cv2.bitwise_or(mask_blue, mask_black) #combinamos ambas mascaras

    return combined_mask    

def segment_object(cap:cv2.VideoCapture, args:argparse)->None:
    # Main loop
    while cap.isOpened():
        # Read current frame
        ret, frame = cap.read()

        # Check if the image was correctly captured
        if not ret:
            print("ERROR! - current frame could not be read")
            break

        # Resize current frame
        frame = rescale_frame(frame, args.frame_resize_percentage)

        binarized_frame = binary_colour(frame)

        x, y, w, h = cv2.boundingRect(binarized_frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
        cv2.putText(frame,'Dr. Andres',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),1,cv2.LINE_AA)

        # Visualise both the input video and the object detection windows
        cv2.imshow(window_params['capture_window_name'], frame) #mostramos el video con el rectangulo
        cv2.imshow(window_params['detection_window_name'], binarized_frame) #mostramos el video binarizado

        # The program finishes if the key 'q' is pressed
        key = cv2.waitKey(5)
        if key == ord('q') or key == 27:
            print("Program finished!")
            break

def close_windows(cap:cv2.VideoCapture)->None:
    
    # Destroy all visualisation windows
    cv2.destroyAllWindows()

    # Destroy 'VideoCapture' object
    cap.release()
