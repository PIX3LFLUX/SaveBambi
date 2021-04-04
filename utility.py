import cv2
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Detection of deer from thermal images")
    parser.add_argument('--path_to_input_video', type=str, help='the path to the input video on which detection of animals have to be done', required=True)
    parser.add_argument('--path_to_output_video',type=str,help='the path to where output video has to be stored',required =True)
    parser.add_argument('--path_to_saved_model',type=str,help='the path to saved ML model to be used',required =True)
    parser.add_argument('--path_to_labelmap',type=str,help='the path to label map',required =True)
    parser.add_argument('--debug',type=str,default=False)
    return parser.parse_args()
