import os
import csv

def get_streetview_coord(coord_folder, destination_folder):

    # Récupérer les noms des sous-dossiers
    sous_dossiers = [d for d in os.listdir(coord_folder) if os.path.isdir(os.path.join(coord_folder, d))]

    # Préparer les en-têtes du fichier CSV
    entetes = ['Longitude', 'Latitude']

    # Créer une liste pour stocker les valeurs de longitude et de latitude
    valeurs = []

    # Parcourir les sous-dossiers et extraire les valeurs de longitude et de latitude
    for sous_dossier in sous_dossiers:
        nom_sous_dossier = sous_dossier.split(",")  # Supposant que les valeurs sont séparées par une virgule
        if len(nom_sous_dossier) == 2:
            longitude = nom_sous_dossier[1].strip()
            latitude = nom_sous_dossier[0].strip()
            valeurs.append([longitude, latitude])

    # Créer le fichier CSV
    
    with open(destination_folder, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(entetes)
        writer.writerows(valeurs)

    print("Le fichier CSV a été créé avec succès.")


if __name__ == "__main__":
    coord_folder = "ForPrediction"
    destination_folder = "streetview_coord_2.csv"

    get_streetview_coord(coord_folder, destination_folder)