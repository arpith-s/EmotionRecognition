import pandas as pd
import generate_dataset as gd

if __name__ == '__main__':
    df0 = pd.read_csv('emotionData.csv')
    er = gd.GenerateDataset(df0)
    er.csv_2img()
