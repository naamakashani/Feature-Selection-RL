import pandas as pd
import numpy as np


def create_None():
    # File path for the original data
    file_pth = 'C:\\Users\\kashann\\PycharmProjects\\choiceMira\\RL\\extra\\diabetes\\diabetes_prediction_dataset.csv'

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_pth)

    # Skip the first row and randomly assign NaN to 20% of the cells in each row
    for i in range(1, len(df)):
        for j in range(len(df.columns)-1):
            if np.random.rand() < 0.2:
                df.iloc[i, j] = np.nan
    #if last column is 1 write 0
    # df['fetal_health'] = [0 if x == 1 else 1 for x in df['fetal_health']]


    # File path for the new data with None values
    output_pth = r'C:\Users\kashann\PycharmProjects\choiceMira\data\diabetes_None.csv'

    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_pth, index=False)



if __name__ == "__main__":
    create_None()
