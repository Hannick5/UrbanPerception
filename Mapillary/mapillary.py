import os
import csv
import requests
from tqdm import tqdm

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

# Create the "centroid_data" directory if it doesn't exist
if not os.path.exists('centroid_data_3'):
    os.makedirs('centroid_data_3')

# Open the CSV file containing centroid coordinates
with open('centroid_3.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)

    # Skip the header row
    next(reader)

    # Calculate the total number of centroids
    total_centroids = sum(1 for _ in reader)

    # Reset the reader position
    csvfile.seek(0)
    next(reader)  # Skip the header row

    # Create a progress bar
    progress_bar = tqdm(total=total_centroids, desc='Processing centroids', unit='centroid')

    # Sequential counter for output file names
    file_counter = 1

    # Iterate over the rows of the CSV file
    for row in reader:
        centroid_longitude = float(row['Longitude'])
        centroid_latitude = float(row['Latitude'])

        # Calculate the bounding box coordinates for the grid square around the centroid
        min_longitude = centroid_longitude - grid_size / 2
        max_longitude = centroid_longitude + grid_size / 2
        min_latitude = centroid_latitude - grid_size / 2
        max_latitude = centroid_latitude + grid_size / 2

        # Retrieve the image metadata within the grid square
        images_metadata = get_images_metadata(min_longitude, min_latitude, max_longitude, max_latitude)

        # Generate the output CSV file path with sequential number
        output_file_path = os.path.join('centroid_data_3', f'metadata_{file_counter}.csv')

        # Open the output CSV file for writing
        with open(output_file_path, 'w', newline='') as outputfile:
            fieldnames = ['ID', 'Longitude', 'Latitude']
            writer = csv.DictWriter(outputfile, fieldnames=fieldnames)

            # Write the header row in the output CSV file
            writer.writeheader()

            # Write the image information to the output CSV file
            for metadata in images_metadata:
                image_id = metadata['id']
                image_longitude = metadata['geometry']['coordinates'][0]
                image_latitude = metadata['geometry']['coordinates'][1]
                writer.writerow({'ID': image_id, 'Longitude': image_longitude, 'Latitude': image_latitude})

        # Update the progress bar and file counter
        progress_bar.update(1)
        file_counter += 1

    # Close the progress bar
    progress_bar.close()
