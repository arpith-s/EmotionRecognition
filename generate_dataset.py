from PIL import Image
import numpy as np
import crayons as cr
import shutil
import os


class GenerateDataset:

    # Initialise variables required.
    def __init__(self, df):
        self.df = df
        self.__parent_dir_list = ['train', 'test']
        self.__child_dir_list = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    def csv_2img(self):
        print()
        if os.path.exists('dataset'):
            if self.get_size() > 6000000:
                print(cr.blue('Dataset already present!\nRegenerate data? (Y/N) :', bold=True), end='')
                a = input().strip().lower()
                if a == 'n':
                    return
                elif a == 'y':
                    shutil.rmtree('dataset/')
                else:
                    print(cr.red("Invalid Input", bold=True))
                    exit(1)

        print(cr.blue('Extracting data...', bold=True))
        try:
            # Create required directories if they don't exist.
            if not os.path.exists('dataset'):
                os.makedirs('dataset')
            for par_dir in self.__parent_dir_list:
                if not os.path.exists('dataset/' + par_dir):
                    os.makedirs('dataset/' + par_dir)
                for child_dir in self.__child_dir_list:
                    if not os.path.exists('dataset/' + par_dir + '/' + child_dir):
                        os.makedirs('dataset/' + par_dir + '/' + child_dir)

            # Paths to store the images extracted from the csv file.
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

            # Iterate through all the rows of the DataFrame.
            for index, row in self.df.iterrows():
                # Get pixel data from the DataFrame
                pixel = str(row['pixels']).split()

                # Reshape the pixel matrix to 48x48.
                data = np.array(pixel).reshape(48, 48)

                # Generate image from pixel data.
                img = Image.fromarray(data.astype('uint8'))

                # Put the image in the appropriate folder according to their emotions mentioned.
                if str(row['Usage']).strip().lower() == 'training':
                    path = data_loc.get(row['emotion'], '')
                else:
                    path = data_loc.get(row['emotion'], '').replace('train', 'test')

                # Save Image
                img.save(path + str(index) + '.png')
            print(cr.blue('Data Extraction Finished!', bold=True))
        except Exception as e:
            print()
            print(cr.red(str(e), bold=True))
            exit(1)

    @staticmethod
    def get_size(start_path=r'dataset/'):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(start_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # skip if it is symbolic link
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        return total_size
