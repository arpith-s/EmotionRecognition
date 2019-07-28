import convolutional_neural_network
import matplotlib.pyplot as plt
import generate_dataset as gd
import data_generator as gen
import pandas as pd
import os


class EmotionRecognition:
    @staticmethod
    def plot_graph(history):
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Loss', fontsize=16)

        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Testing Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)

        plt.plot(history.history['acc'], label='Training Accuracy')
        plt.plot(history.history['val_acc'], label='Testing Accuracy')
        plt.legend()

        plt.show()


if __name__ == '__main__':
    emotion_recognition = EmotionRecognition()
    df = pd.read_csv('emotionData.csv')

    er = gd.GenerateDataset(df)
    er.csv_2img()

    train_generator, test_generator = gen.PrepareData()

    model = convolutional_neural_network.CNN(train_generator, test_generator)
    if os.path.isfile('model/CNN.h5'):
        print('Model Already Exists! Do you want to retrain the model? (Y/N): ')
        if input().lower().strip() == 'y':
            history = model.create()
            print('Do you want to plot the analysis graphs? (Y/N): ')
            if input().lower().strip() == 'y':
                emotion_recognition.plot_graph(history)
    else:
        print('Training Model')
        history = model.create()
        print('Do you want to plot the analysis graphs? (Y/N): ')
        if input().lower().strip() == 'y':
            emotion_recognition.plot_graph(history)
