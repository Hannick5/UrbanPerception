import csv
from collections import Counter

def count_duplicate_hubnames(csv_file):
    hubname_counter = Counter()
    
    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        
        for row in csv_reader:
            hubname = row['HubName']
            hubname_counter[hubname] += 1
    
    duplicate_count = sum(count > 1 for count in hubname_counter.values())
    return duplicate_count


csv_file = 'test.csv'
duplicate_count = count_duplicate_hubnames(csv_file)
print(f"Number of duplicate HubNames: {duplicate_count}")
