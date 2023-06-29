import os
import csv

def count_csv_files_with_header_only(folder_path):
    count = 0

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                csv_reader = csv.reader(file)
                header = next(csv_reader, None)  # Read the header

                # Check if the header is not None and there are no other rows
                if header is not None and next(csv_reader, None) is None:
                    count += 1

    return count

if __name__ == "__main__":
    
    folder_path = 'centroid_data_3'
    count = count_csv_files_with_header_only(folder_path)
    print(f"Number of CSV files with only a header: {count}")
