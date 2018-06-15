# MotionPrediction
Hand recognition with Motion Prediction

gesture-32data.save: https://drive.google.com/file/d/1klVnOfdhcI7j8raNxoBS6VehWIu_KsCp/view?usp=sharing
test-32data.save: https://drive.google.com/open?id=1KOGMOF8CQX6EjPoj6HMT1kx8Gh6um-6S
test-32data-fastward-onlylabel.save: https://drive.google.com/open?id=1gVuiHKFY_PkyVgGgOKvWD3XwNzYvDHYh

Files included:
  1. hand pose estimation with prediction.ipynb
     This ipynb file is used for data importing, gesture prediction, data validation and evaluation.

  2. hand pose estimation.ipynb
     This ipynb file is used for data importing, model training and hand pose estimation

  3. hand gesture recognition.ipynb
  	 This ipynb file is used for recognize the gestures in images

  4. data.py
     This script is used for data processing part, it can pick up the hands part out of the image and compress the images resolution from 1080P to 64 * 64 and 32 * 32.

  5. Gesture.py 
     This is a python script we use to label the picture with gesture status and stored into Json format.
