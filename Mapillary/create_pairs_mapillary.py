import csv

def create_pair_csv(file1, file2, output_file):
    # Chargement des données du premier fichier CSV (longitude/latitude)
    data1 = []
    with open(file1, 'r') as f1:
        reader = csv.DictReader(f1)
        for row in reader:
            data1.append(row)

    # Chargement des données du deuxième fichier CSV (HubName)
    data2 = {}
    with open(file2, 'r') as f2:
        reader = csv.DictReader(f2)
        for row in reader:
            hub_name = row['HubName']
            longitude = float(row['Longitude'])
            latitude = float(row['Latitude'])
            data2[(longitude, latitude)] = hub_name

    # Création du nouveau fichier CSV
    with open(output_file, 'w', newline='') as output_f:
        fieldnames = ['HubName1', 'HubName2', 'Label']
        writer = csv.DictWriter(output_f, fieldnames=fieldnames)
        writer.writeheader()

        # Parcours des enregistrements du premier fichier et création des paires
        for i in range(len(data1)):
            longitude1 = float(data1[i]['Longitude1'])
            latitude1 = float(data1[i]['Latitude1'])
            longitude2 = float(data1[i]['Longitude2'])
            latitude2 = float(data1[i]['Latitude2'])

            hub_name1 = data2[(longitude1, latitude1)]
            hub_name2 = data2[(longitude2, latitude2)]
            label = data1[i]['Label']

            writer.writerow({'HubName1': hub_name1, 'HubName2': hub_name2, 'Label': label})

    print("Le nouveau fichier CSV a été créé avec succès.")

create_pair_csv("output.csv", "mapillary_correct_streetview.csv", "final_training_query_mapillary.csv")