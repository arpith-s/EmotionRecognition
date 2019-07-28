import pandas as pd
import numpy as np
from PIL import Image
import os


class GenerateDataset:
    def __init__(self, df):
        self.df = df
        self.__parent_dir_list = ['train', 'test']
        self.__child_dir_list = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    def csv_2img(self):

        if not os.path.exists('dataset'):
            os.makedirs('dataset')

        for par_dir in self.__parent_dir_list:
            if not os.path.exists('image/' + par_dir):
                os.makedirs('image/' + par_dir)
            for child_dir in self.__child_dir_list:
                if not os.path.exists('image/' + par_dir + '/' + child_dir):
                    os.makedirs('image/' + par_dir + '/' + child_dir)

        # 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
        data_loc = {
            0: 'dataset/train/angry/',
            1: 'dataset/train/disgust/',
            2: 'dataset/train/fear/',
            3: 'dataset/train/happy/',
            4: 'dataset/train/sad/',
            5: 'dataset/train/surprise/',
            6: 'dataset/train/neutral/'
        }

        for index, row in self.df.iterrows():
            pixel = str(row['pixels']).split()

            data = np.array(pixel).reshape(48, 48)

            img = Image.fromarray(data.astype('uint8'))

            if str(row['Usage']).strip().lower() == 'training':
                path = data_loc.get(row['emotion'], '')
            else:
                path = data_loc.get(row['emotion'], '').replace('train', 'validation')

            img.save(path + str(index) + '.png')


if __name__ == '__main__':
    df0 = pd.read_csv('emotionData.csv')

    er = GenerateDataset(df0)
    # er.csv_2img()
