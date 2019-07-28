import pandas as pd
import numpy as np
from PIL import Image


class GenerateDataset:
    def __init__(self, df):
        self.df = df

    def csv_2img(self):

        # 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
        data_loc = {
            0: 'images/train/angry/',
            1: 'images/train/disgust/',
            2: 'images/train/fear/',
            3: 'images/train/happy/',
            4: 'images/train/sad/',
            5: 'images/train/surprise/',
            6: 'images/train/neutral/'
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
