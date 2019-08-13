# EmotionRecognition  [![GitHub release](https://img.shields.io/badge/Release-v1-green.svg?&colorA=024a70&?&colorB=0779b5)](https://github.com/Arpith-kumar/EmotionRecognition)&nbsp; [![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg?logo=paypal)](https://www.paypal.me/arpith09)&nbsp; [![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg?logo=linkedin)](https://www.linkedin.com/in/arpith-s/)&nbsp; 
<!-- [![Support](https://www.buymeacoffee.com/assets/img/custom_images/yellow_img.png)](https://www.buymeacoffee.com/arpith) -->


Emotion recognition (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) using CNN with Python 3.x

![](https://github.com/Arpith-kumar/EmotionRecognition/blob/master/Image/ezgif.com-video-to-gif.gif)

EmotionRecognition in action.


## Usage:

  1. Move to the project directory.
  
  2. Make sure you have all the libraries required by running the following command on your cmd/terminal.
     
     ```
     pip install -r requirements.txt
     ```
  
  3. The dataset to train and test our CNN model will be extracted from a CSV file with the following description:
        * The data consists of 48x48 pixel grayscale images of faces. 
        * The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. 
        * The task is to categorize each face based on the emotion shown in the facial expression into one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
        * emotionData.csv contains two columns, "emotion" and "pixels". 
        * The "emotion" column contains a numeric code ranging from 0 to 6, inclusive, for the emotion that is present in the image. 
        * The "pixels" column contains a string surrounded in quotes for each image. 
        * The contents of this string a space-separated pixel values in row-major order.
        
          * [emotionData.csv](https://drive.google.com/file/d/1dZESuROeSwyAUB31Ckbbmc3KM_mnkOER/view?usp=sharing) - Download dataset file
  
  4. This program uses your phone's camera as its video source. To make that work you'll have to download the IP Webcam application from the Google Play Store. 
     After you have the application downloaded scroll to the bottom and press Start Server which begins to stream the video. 
     Once the video starts to stream you will see the IPv4 address displayed on the screen of the mobile application.
     Copy that IP address and go to main.py -> EmotionRecognition class -> init and set the URL to the IP Address mentioned on IP Webcam application. 
     Note: Your PC and mobile should be connected to the same wi-fi network for this to work.
     
      * [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam&hl=en_IN) - Download IP Webcam
  
  5. Finally, run the main.py file to start execution.
     
     Note: Make sure the emotionData.csv dataset file is in the project directory and also make sure the video is streaming before running main.py [4]
      
      ```
      python main.py
      ```
## Output:

![alt text](https://github.com/Arpith-kumar/EmotionRecognition/blob/master/Image/1.png)

![alt text](https://github.com/Arpith-kumar/EmotionRecognition/blob/master/Image/2.png)

![alt text](https://github.com/Arpith-kumar/EmotionRecognition/blob/master/Image/3.png)

![alt text](https://github.com/Arpith-kumar/EmotionRecognition/blob/master/Image/4.png)

![alt text](https://github.com/Arpith-kumar/EmotionRecognition/blob/master/Image/5.png)

## Built With:

* [Keras](https://keras.io/) - The Python Deep Learning library.
* [Pandas](https://pandas.pydata.org/) - Python Data Analysis Library.
* [NumPy](https://www.numpy.org/) - The fundamental package for scientific computing with Python.
* [OpenCV](https://opencv.org/) - A library of programming functions mainly aimed at real-time computer vision.
* [PIL](https://pillow.readthedocs.io/en/stable/) - Python Imaging Library is a free library for the Python programming language that adds support for opening, manipulating, and saving many different image file formats.
* [Crayons](https://pypi.org/project/crayons/) - TextUI colors for Python.