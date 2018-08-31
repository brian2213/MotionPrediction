# MotionPrediction
Hand Pose Estimation and Hand recognition with Motion Prediction

**Author**: YiMing Chen<br /><br />
**Author**: Hengchen Zhou<br /><br />
**Author**: Mingyu Jiang<br /><br />

## Files
  - **hand pose estimation with prediction.ipynb**:
     This ipynb file is used for data importing, gesture prediction, data validation and evaluation.

  - **hand pose estimation.ipynb**:
     This ipynb file is used for data importing, model training and hand pose estimation

  - **hand gesture recognition.ipynb**:
  	 This ipynb file is used for recognize the gestures in images
   
  - **hand gesture recognition2.ipynb**:
     This ipynb file combine the pose estimation model with the gesture recgnition model to judge the current gesture in image.

  - **data.py**:
     This script is used for data processing part, it can pick up the hands part out of the image and compress the images resolution from 1080P to 64 * 64 and 32 * 32.

  - **gesture_label.py**:
     This is a python script we use to label the picture with gesture status and stored into Json format.
     
  - **gesture_data.py**:
     This is a python script we use to combine the processed image file with the gesture label json file to produce the data file for          gesture recognition training and testing.
     
# To run the model

All the working scrips are in the Jupyter folder, please go straight to the Jupyter folder to `execute  *.ipynb`.

You need to download below `all the *.save data` files and put them in the same directory the *.ipynb files in (/Jupyter).

- **gesture-32data.save**: https://drive.google.com/file/d/1klVnOfdhcI7j8raNxoBS6VehWIu_KsCp/view?usp=sharing
- **test-32data.save**: https://drive.google.com/open?id=1KOGMOF8CQX6EjPoj6HMT1kx8Gh6um-6S
- **test-32data-fastward-onlylabel.save**: https://drive.google.com/open?id=1gVuiHKFY_PkyVgGgOKvWD3XwNzYvDHYh

We develop and debug our project on the Google Colab environment.

All the file is written in python3
