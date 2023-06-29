import os
import numpy as np
import csv

def extraire_longitudes_latitudes(dossier, delimiteur, fichier_csv):
    with open(fichier_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['ID', 'Longitude', 'Latitude'])  # Écrire l'en-tête du fichier CSV

        for fichier in os.listdir(dossier):
            chemin_fichier = os.path.join(dossier, fichier)
            if os.path.isfile(chemin_fichier):
                nom_fichier, extension = os.path.splitext(fichier)
                valeurs = nom_fichier.split(delimiteur)
                if len(valeurs) >= 2:
                    try:
                        longitude = float(valeurs[0])
                        latitude = float(valeurs[1])
                        for i in range(2, len(valeurs)):
                            if len(str(valeurs[i])) > 4:
                                id = str(valeurs[i])
                                break
                        writer.writerow([id, longitude, latitude])
    
                    except ValueError:
                        continue

# Utilisation de la fonction
dossier = "data/question_1/Sample_web_green"
delimiteur = "_"
fichier_csv = "fichier.csv"

extraire_longitudes_latitudes(dossier, delimiteur, fichier_csv)
