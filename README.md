# DriverDrowsiness_CNN_LSTM

The code is developed based on the research paper https://www.researchgate.net/publication/383227527_A_CNN-LSTM_APPROACH_FOR_ACCURATE_DROWSINESS_AND_DISTRACTION_DETECTION_IN_DRIVERS for Driver drowsiness detection.

Mediapipe facelandmark model is used to extract 478 face landmarks to extract the features like.

1. R-EAR,
2. L-EAR,
3. MAR, 
4. HeadPose

![image](https://github.com/user-attachments/assets/b4774ecf-cee0-4617-a3df-991685f581a5)

Head pose is calculated using Perspective-n-Point (PnP) method 

The collected data is stacked rowsise and split into window of 16 frames.
Below is the graph that shows the values for a single video.

![image](https://github.com/user-attachments/assets/850adb42-5cf6-4ed8-b7c8-9f5174d8ef06)

The dataset is collected similar to YawDDDatset for 3 classes Normal,Eye close and Yawning.

Below is the Folder structure.

Customdataset
    - Normal
    - Eye close
    - Yawning
        - Yawning_frame0.avi
        - Yawning_frame1.avi

To train on existing preprocessing,

In config.ini set use_existing_preprocessesd_data = True

Or to train on new dataset

In config.ini set use_existing_preprocessesd_data = False and the datapath

Exceute python train.py

To run inference,

Execute python inference.py
