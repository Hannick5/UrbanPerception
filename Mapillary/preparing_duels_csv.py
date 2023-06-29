import os
import csv
import pandas as pd
from tqdm import tqdm

def prepare_csv(csv_file, csv_file_2, output_csv):
    headers1 = ['Id1', 'Id2', 'label', "_"]

    df1 = pd.read_csv(csv_file, names=headers1)
    df2 = pd.read_csv(csv_file_2)

    # Initialize lists for the coordinates
    longitude1 = []
    latitude1 = []
    longitude2 = []
    latitude2 = []

    # Set up the tqdm progress bar
    pbar = tqdm(total=len(df1), desc="Processing", unit=" row")

    for index1, row1 in df1.iterrows():
        for index2, row2 in df2.iterrows():
            if row1["Id1"] in row2["Image Name"]:
                # Split the "Image Name" with the delimiter "_"
                image_name_parts = row2["Image Name"].split("_")
                # Extract the longitude and latitude values
                longitude1.append(image_name_parts[0])
                latitude1.append(image_name_parts[1])

            if row1["Id2"] in row2["Image Name"]:
                # Split the "Image Name" with the delimiter "_"
                image_name_parts = row2["Image Name"].split("_")
                # Extract the longitude and latitude values
                longitude2.append(image_name_parts[0])
                latitude2.append(image_name_parts[1])

        # Update the progress bar
        pbar.update(1)

    # Close the progress bar
    pbar.close()

    # Create a new DataFrame with the extracted coordinates
    result_df = pd.DataFrame({
        'Id1': df1['Id1'],
        'Id2': df1['Id2'],
        'Label': df1['label'],
        'Longitude1': longitude1,
        'Latitude1': latitude1,
        'Longitude2': longitude2,
        'Latitude2': latitude2
    })

    # Save the DataFrame to a CSV file
    result_df.to_csv(output_csv, index=False)

    print("CSV file '{}' has been created.".format(output_csv))


            
csv_file = 'data/question_1/duels_question_1.csv'
csv_file_2 = 'Mapillary/train_image_names.csv'
output_csv = 'output.csv'

prepare_csv(csv_file, csv_file_2, output_csv)

