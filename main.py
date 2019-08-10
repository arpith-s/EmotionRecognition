import convolutional_neural_network
from keras.models import load_model
import matplotlib.pyplot as plt
import generate_dataset as gd
import data_generator as gen
import pandas as pd
import numpy as np
import requests
import crayons as cr
import cv2
import os


class EmotionRecognition:

    def __init__(self):
        self.emotions_list = ["ANGRY", "DISGUSTED", "FEAR", "HAPPY",
                              "NEUTRAL", "SAD", "SURPRISED"]
        self.classifier = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
        self.url = 'http://192.168.1.3:8080/shot.jpg'

    @staticmethod
    def load():
        return load_model('models/CNN.h5')

    def predict_emotion(self, img, new_model):
        prediction = new_model.predict(img)
        emotions = self.emotions_list[np.argmax(prediction)]
        return emotions

    # returns camera frames along with bounding boxes and predictions
    def stream_video(self, m):

        while True:
            img_resp = requests.get(self.url)
            img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
            frame = cv2.imdecode(img_arr, -1)

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.classifier.detectMultiScale(frame_gray, 1.3, 5)

            if str(type(faces)) == str(type(())):
                cv2.imshow("Live Stream", frame)
                if cv2.waitKey(1) == 27:
                    break
                continue

            for (x, y, w, h) in faces:
                crop_frame = frame_gray[y:y + h, x:x + w]

                image = cv2.resize(crop_frame, (48, 48))
                pred = self.predict_emotion(image[np.newaxis, :, :, np.newaxis], m)

                cv2.putText(frame, pred, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                cv2.imshow("Live Stream", frame)

            if cv2.waitKey(1) == 27:
                break

        # cleanup the camera and close any open windows
        cv2.destroyAllWindows()

    @staticmethod
    def plot_graph(history0):
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)

        plt.plot(history0.history['loss'], label='Training Loss')
        plt.plot(history0.history['val_loss'], label='Testing Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)

        plt.plot(history0.history['acc'], label='Training Accuracy')
        plt.plot(history0.history['val_acc'], label='Testing Accuracy')
        plt.legend()

        plt.show()


if __name__ == '__main__':

    try:
        df = pd.read_csv('emotionData.csv')

        er = gd.GenerateDataset(df)
        er.csv_2img()

        train_generator, test_generator = gen.PrepareData().run()

        model = convolutional_neural_network.CNN(train_generator, test_generator)

        emotion_recognition = EmotionRecognition()

        if not os.path.exists('models'):
            os.makedirs('models')

        if os.path.isfile('models/CNN.h5'):
            print(cr.blue('Model Already Exists! Do you want to retrain the model? (Y/N): ', bold=True), end='')
            y = input().lower().strip()
            if y == 'y':
                os.remove('models/CNN.h5')
                history = model.create()
                print(cr.blue('Do you want to plot the analysis graphs? (Y/N): ', bold=True), end='', )
                x = input().lower().strip()
                if x == 'y':
                    emotion_recognition.plot_graph(history)
                elif x == 'n':
                    pass
                else:
                    print(cr.red("Invalid Input", bold=True))
                    exit(1)
            elif y == 'n':
                pass
            else:
                print(cr.red("Invalid Input", bold=True))
                exit(1)

        else:
            print(cr.blue('Training Model...', bold=True))
            history = model.create()
            print(cr.blue('Do you want to plot the analysis graphs? (Y/N): ', bold=True), end='0')
            a = input().lower().strip()
            if a == 'y':
                emotion_recognition.plot_graph(history)
            elif a == 'n':
                pass
            else:
                print(cr.red("Invalid Input", bold=True))
                exit(1)
        emotion_recognition.stream_video(emotion_recognition.load())
    except Exception as e:
        print()
        print(cr.red(str(e), bold=True))
        exit(1)
