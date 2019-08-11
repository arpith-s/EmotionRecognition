# EmotionRecognition ![](<style>.bmc-button img{width: 27px !important;margin-bottom: 1px !important;box-shadow: none !important;border: none !important;vertical-align: middle !important;}.bmc-button{line-height: 36px !important;height:37px !important;text-decoration: none !important;display:inline-flex !important;color:#ffffff !important;background-color:#FF813F !important;border-radius: 3px !important;border: 1px solid transparent !important;padding: 1px 9px !important;font-size: 22px !important;letter-spacing:0.6px !important;box-shadow: 0px 1px 2px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 1px 2px 2px rgba(190, 190, 190, 0.5) !important;margin: 0 auto !important;font-family:'Cookie', cursive !important;-webkit-box-sizing: border-box !important;box-sizing: border-box !important;-o-transition: 0.3s all linear !important;-webkit-transition: 0.3s all linear !important;-moz-transition: 0.3s all linear !important;-ms-transition: 0.3s all linear !important;transition: 0.3s all linear !important;}.bmc-button:hover, .bmc-button:active, .bmc-button:focus {-webkit-box-shadow: 0px 1px 2px 2px rgba(190, 190, 190, 0.5) !important;text-decoration: none !important;box-shadow: 0px 1px 2px 2px rgba(190, 190, 190, 0.5) !important;opacity: 0.85 !important;color:#ffffff !important;}</style><link href="https://fonts.googleapis.com/css?family=Cookie" rel="stylesheet"><a class="bmc-button" target="_blank" href="https://www.buymeacoffee.com/lIjetvg"><img src="https://bmc-cdn.nyc3.digitaloceanspaces.com/BMC-button-images/BMC-btn-logo.svg" alt="Buy me a coffee"><span style="margin-left:5px">Buy me a coffee</span></a>)
Emotion recognition (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) using CNN with Python 3.x

![](https://github.com/Arpith-kumar/EmotionRecognition/blob/master/Image/ezgif.com-video-to-gif.gif)

EmotionRecognition in action.


## Usage:

  1. Move to the project directory.
  
  2. Make sure you have all the librarys required by running the following command on your cmd/terminal.
     
     ```
     pip install -r requirements.txt
     ```
  
  3. The dataset to train and test our CNN model will be extracted from a csv file with the following description:
        * The data consists of 48x48 pixel grayscale images of faces. 
        * The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. 
        * The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
        * emotionData.csv contains two columns, "emotion" and "pixels". 
        * The "emotion" column contains a numeric code ranging from 0 to 6, inclusive, for the emotion that is present in the image. 
        * The "pixels" column contains a string surrounded in quotes for each image. 
        * The contents of this string a space-separated pixel values in row major order.
        
          * [emotionData.csv](https://drive.google.com/file/d/1dZESuROeSwyAUB31Ckbbmc3KM_mnkOER/view?usp=sharing) - Download dataset file
  
  4. This program uses your phone's camara as its video source. To make that work you'll have to download the IP Webcam application from the Google Play Store. 
     After you have the application downloaded scroll to the bottom and press Start Server wich begins to stream the video. 
     Once the video starts to stream you will see the IPv4 address displayed on the screen of the mobile application.
     Copy that IP address and go to main.py -> EmotionRecognition class -> init and set the url to the IP Address mentioned on IP Webcam application. 
     Note: Your PC and mobile should be connected to the same wi-fi network for this to work.
     
      * [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam&hl=en_IN) - Download IP Webcam
  
  5. Finally run the main.py file to start execution.
      
      ```
      python main.py
      ```
## Output:

![alt text](https://github.com/Arpith-kumar/EmotionRecognition/blob/master/Image/1.png)

![alt text](https://github.com/Arpith-kumar/EmotionRecognition/blob/master/Image/2.png)

![alt text](https://github.com/Arpith-kumar/EmotionRecognition/blob/master/Image/3.png)

![alt text](https://github.com/Arpith-kumar/EmotionRecognition/blob/master/Image/4.png)

![alt text](https://github.com/Arpith-kumar/EmotionRecognition/blob/master/Image/5.png)

![alt text](https://github.com/Arpith-kumar/EmotionRecognition/blob/master/Image/1.png)

## Built With:

* [Keras](https://keras.io/) - The Python Deep Learning library.
* [Pandas](https://pandas.pydata.org/) - Python Data Analysis Library.
* [NumPy](https://www.numpy.org/) - The fundamental package for scientific computing with Python.
* [OpenCV](https://opencv.org/) - A library of programming functions mainly aimed at real-time computer vision.
* [PIL](https://pillow.readthedocs.io/en/stable/) - Python Imaging Library is a free library for the Python programming language that adds support for opening, manipulating, and saving many different image file formats.
* [Crayons](https://pypi.org/project/crayons/) - TextUI colors for Python.