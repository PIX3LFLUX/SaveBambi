import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import warnings
import time as time
from utility import parse_args
import cv2 as cv2
warnings.filterwarnings('ignore')

import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


class SaveBambi:

    # init method or constructor
    def __init__(self,path_to_input_video,path_to_labelmap,path_to_output_video,path_to_saved_model,debug):


      
      self.path_to_saved_model=path_to_saved_model
      self.path_to_labelmap=path_to_labelmap
      self.debug=debug
      self.detect_fn= self.load_ml_model()
      self.category_index= self.load_labelmap()
      print(self.detect_fn)
      

      #Setting the video writer and opening the VideoFile
      filename = path_to_output_video
      codec = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
      cap = cv2.VideoCapture(path_to_input_video)
      framerate = round(cap.get(5),2)
      w = int(cap.get(3))
      h = int(cap.get(4))
      resolution = (w, h)
      count=0
      VideoFileOutput = cv2.VideoWriter(filename, codec, framerate, resolution)
      if args.debug:
        print("VideoFile output "+ str(VideoFileOutput) )

      frame_read, image = cap.read()

      # Iterate over frames and pass each for prediction
      if(cap.isOpened()):
          print("Video opened- Processing Started")
     
      frame_read, image = cap.read()
    # Iterate over frames and pass each for prediction
      while frame_read:

        # First checks for presence of an animal
        animal_detected = self.presence_detection(image)
        
        #If animal is detected, pass it to ML model
        if animal_detected:
            image_after_processing=self.run_detection_using_ML(image)

        #if no animal pass the image without further processing
        else:
          
          image_after_processing= image 

        # Read next frame
        frame_read, image = cap.read()
        count += 1

        # Write frame with predictions to video
        VideoFileOutput.write(image_after_processing) 

      #  Release video file when we're ready
      VideoFileOutput.release()
      cap.release()
      print("Done!") 


    def load_ml_model(self):
      """Load the machine learning model trained on thermal images of animals.

        Args:
          path_to_saved_model: the file path to the saved model
          debug: debug flag
        Returns:
          an instance of detect_fn
      """
      if self.debug:

        print('Loading model...', end='')
        print(self.path_to_saved_model)
        time_load=time.time()
     
        detect_fn=tf.saved_model.load(self.path_to_saved_model)
        print(detect_fn)
        print("Time taken to load the model "+ str(time.time()-time_load))
        print('Done!')
      else:
        detect_fn=tf.saved_model.load(self.path_to_saved_model)
        
        
      return detect_fn

    def load_labelmap(self):
      """Loads the label map

            Args:
              path_to_saved_model: the file path to the saved model
              label map is assumed to be placed in the same folder as saved model

            Returns:
              an instance of category_index - label map
      """
      category_index=label_map_util.create_category_index_from_labelmap(self.path_to_labelmap,use_display_name=True)
      return category_index

    def load_image_into_numpy_array(self,image):
      """Load an image from file into a numpy array.

        Puts image into numpy array to feed into tensorflow graph.
        Note that by convention we put it into a numpy array with shape
        (height, width, channels), where channels=3 for RGB.

        Args:
          path: the file path to the image

        Returns:
          uint8 numpy array with shape (img_height, img_width, 3)
      """
        #return np.array(path)
        #img_data = tf.io.gfile.GFile(path, 'rb').read()
        #image = Image.open(BytesIO(img_data))
      im_width, im_height = image.size
      return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


    def run_detection_using_ML(self,image_np):


      """Runs detection and visualises output using ML model

              Args:
                image_np:the input image
                detect_fn : instance of detection function with the ML model
                category_index: label map to be used for visualisation
                debug: debug flag
              Returns:
                image_np_with_detections - images with detections drawn
      """

      input_tensor=tf.convert_to_tensor(image_np)
      # The model expects a batch of images, so add an axis with `tf.newaxis`.
      input_tensor=input_tensor[tf.newaxis, ...]
      # input_tensor = np.expand_dims(image_np, 0)
      if self.debug:
        time_pred=time.time()

      detections=self.detect_fn(input_tensor)

      if self.debug:
        print("Time to predict "+str((time.time()-time_pred)))

      num_detections=int(detections.pop('num_detections'))
      detections={key:value[0,:num_detections].numpy()
                      for key,value in detections.items()}
      detections['num_detections']=num_detections

        # detection_classes should be ints.
      detections['detection_classes']=detections['detection_classes'].astype(np.int64)

      image_np_with_detections=image_np.copy()
      if self.debug:
        time_visualizeboxes=time.time()
      viz_utils.visualize_boxes_and_labels_on_image_array(
              image_np_with_detections,
              detections['detection_boxes'],
              detections['detection_classes'],
              detections['detection_scores'],
              self.category_index,
              use_normalized_coordinates=True,
              max_boxes_to_draw=1,     #max number of bounding boxes in the image
              min_score_thresh=.3,      #min prediction threshold
              agnostic_mode=False)
      if self.debug:
        print("time to generate boxes in a frame = " + str(time.time() - time_visualizeboxes))

      return image_np_with_detections



    def presence_detection(self,input_image):
      """checks for presence of an animal using thresholding
        Args:
          input_image - image to be checked
        Returns:
          animal_detected - a flag which is set as True if animal is detected.
      """

      AREA_THRESHOLD=200
      animal_detected=False #resetting animal_detected flag

      imgray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

      #Gaussian filter
      blur = cv2.GaussianBlur(imgray,(51,51),sigmaX=0.5)

      # global thresholding with blur
      ret1,thresh = cv2.threshold(imgray,160,255,cv2.THRESH_BINARY)


      #morphological operations
      kernel = np.ones((3, 3), np.uint8)
      thresh = cv2.erode(thresh, kernel, iterations=1)
      kernel = np.ones((2, 2), np.uint8)
      thresh = cv2.dilate(thresh, kernel, iterations=3)
      

      #next step is to find the contours of the image
      contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
       #checking for areas of the contour if it's greater than AREA_THRESHOLD
      for c in contours:
         area=cv2.contourArea(c)
         
         if area>=AREA_THRESHOLD:
           animal_detected=True


      return animal_detected


    def detect_animal(image):

      """checks for presence of an animal and if found return the image frame
            with detected animal labeled and marked with boundary boxes

        Args:
          input_image - image to be checked
          detect_fn- detection function instance using ML model to be used
          category_index - label map to be used
          debug-debug file

        Returns:
          image_after_processing - image after processing
      """

      #first checking for presence of an animal
      animal_detected = presence_detection(image)

      if animal_detected:
          image_after_processing=run_detection_using_ML(image,detect_fn,category_index,debug)

      else: #if no animal pass the image without further processing
          image_after_processing= image

      return image_after_processing


if __name__ == '__main__':

  args = parse_args()
  SaveBambi(args.path_to_input_video,args.path_to_labelmap,args.path_to_output_video,args.path_to_saved_model,args.debug)
