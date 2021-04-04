
#SaveBambi
A Project to Detect Animals during the Mowing Process - Especially to Protect Roe Deer Fawns

**Abstract**:
Fields in Germany are mowed at different times of the year. Nowadays, this mowing process is often carried out using agricultural machinery with different types of mow-ers. During these mowing operations various animals (insects, birds, amphibians, wild mammals) can be injured or killed. Fawns are particularly affected, as they remain by instinct, motionless in the grass even when in the face of danger. The present techno-logical solution for this problem is to search the fields before the mowing, by use of infrared cameras on drones and deterring by means of acoustic signals. However, these methods require preplanned communication before and during the search, and extra manpower. Hence there is a need for an automated system that can be mounted on the tractor and can detect the animal searched and warn the operator without any manual intervention. The aim of this project is to detect presence of animals in thermal images and if found localize their position in the box using bounding boxes, that can be used in such an automated system. Here detection of all kinds of warm bodied animals are considered, even though roe deer fawns are more prone to threat.

The detection of the animal in thermal images is done in two steps. First check presence of the animal using thresholding. If found, pass the image to a CNN model trained on thermal images for better detction and visualisation using boundary boxes.

**How to use this project?**
 
 Follow the jupyter notebook Automatic detection of roe deer fawns from IR images.ipynb for installing the dependencies.
 Clone this repository and unzip TrainedModel.zip.
 Run the following file to get detections on an input video:
 ```
 !python SaveBambi.py --path_to_input_video=PATH_TO_INPUT_VIDEO 
                          --path_to_output_video=PATH_TO_OUTPUT_VIDEO 
                          --path_to_labelmap=PATH_TO_LABEL_MAP   
                          --path_to_saved_model=PATH_TO_SAVED_MAP
  ```
  Note: Pls remember to correctly give the path to the input and saved model correctly.
  here *path_to_input_video -> path to the input video on which detections have to be made
       *path_to_output_video -> path to where output file must be stored
       *path_to_labelmap -> path to the label map used in the ML model
       *path_to_saved_model -> path to the ML model trained on thermal images
       
**Example detection:**
![image](https://user-images.githubusercontent.com/69155972/111621491-58fbee00-87e8-11eb-99e3-b8d1af7ff0b4.png)


**DataSet**
The dataset used for training the model is present in the folder Dataset.
To add more images, use LabelImg to annotate the images and prepare the data first. 
Refer the folder Dataprepration for more details.

**Training**
To train a new model, please use the jupyter notebook 'Training of a model using TF OF API.ipynb'.
