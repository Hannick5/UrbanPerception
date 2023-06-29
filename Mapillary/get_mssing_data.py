import os
import csv
import requests

# Replace these values with your own information
mly_key = 'MLY|6185142121541114|b499ccd56295db0bff9704d2408e1b90'
grid_size = 0.01  # Grid square size in degrees

# Function to retrieve image metadata within a bounding box
def get_images_metadata(min_longitude, min_latitude, max_longitude, max_latitude):
    bbox_str = f'{min_longitude},{min_latitude},{max_longitude},{max_latitude}'
    url = f'https://graph.mapillary.com/images?access_token={mly_key}&bbox={bbox_str}'
    response = requests.get(url)
    data = response.json()
    features = data.get('data', [])
    return features

# Create the "missing_data" directory if it doesn't exist
if not os.path.exists('missing_data'):
    os.makedirs('missing_data')

# Function to check if a CSV file contains only headers
def is_csv_empty(file_path):
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader, None)  # Read the header

        # Check if the header is not None and there are no other rows
        if header is not None and next(csv_reader, None) is None:
            return True

    return False

# Function to retrieve centroid coordinates from centroids.csv based on index
def get_centroid_coordinates(index):
    with open('centroid_3.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row

        # Find the centroid at the specified index
        for i, row in enumerate(reader):
            if i == index:
                centroid_longitude = float(row[0])
                centroid_latitude = float(row[1])
                return {'Longitude': centroid_longitude, 'Latitude': centroid_latitude}

    return None

# Get the list of CSV files in the "centroid_data" directory
csv_files = [file for file in os.listdir('centroid_data_3') if file.endswith('.csv')]

# Iterate over CSV files
for csv_file in csv_files:
    file_path = os.path.join('centroid_data_3', csv_file)
    if is_csv_empty(file_path):
        # Extract centroid index from the file name
        centroid_index = int(csv_file.split('_')[1].split('.')[0])

        # Retrieve centroid coordinates based on the index
        centroid = get_centroid_coordinates(centroid_index)

        if centroid is not None:
            # Calculate the bounding box coordinates for the grid square around the centroid
            centroid_longitude = float(centroid['Longitude'])
            centroid_latitude = float(centroid['Latitude'])
            min_longitude = centroid_longitude - grid_size / 2
            max_longitude = centroid_longitude + grid_size / 2
            min_latitude = centroid_latitude - grid_size / 2
            max_latitude = centroid_latitude + grid_size / 2

            # Retrieve the image metadata within the grid square
            images_metadata = get_images_metadata(min_longitude, min_latitude, max_longitude, max_latitude)

            # Generate the output CSV file path
            output_file_path = os.path.join('missing_data', csv_file)

            # Write the image metadata to the output CSV file
            with open(output_file_path, 'w', newline='') as output_file:
                fieldnames = ['ID', 'Longitude', 'Latitude']
                writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                writer.writeheader()
                for metadata in images_metadata:
                    image_id = metadata['id']
                    image_longitude = metadata['geometry']['coordinates'][0]
                    image_latitude = metadata['geometry']['coordinates'][1]
                    writer.writerow({'ID': image_id, 'Longitude': image_longitude, 'Latitude': image_latitude})
