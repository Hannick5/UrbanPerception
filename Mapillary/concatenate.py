import os
import pandas as pd

def concatener_csv_dossier(dossier, fichier_sortie):
    # Récupérer la liste des fichiers CSV dans le dossier
    fichiers_csv = [fichier for fichier in os.listdir(dossier) if fichier.endswith(".csv")]

    # Créer une liste pour stocker les données de chaque fichier CSV
    donnees_concatenees = []

    # Parcourir chaque fichier CSV et le concaténer aux données
    for fichier_csv in fichiers_csv:
        chemin_fichier = os.path.join(dossier, fichier_csv)
        donnees_csv = pd.read_csv(chemin_fichier)
        donnees_concatenees.append(donnees_csv)

    # Concaténer toutes les données en un seul DataFrame
    donnees_concatenees = pd.concat(donnees_concatenees, ignore_index=True)

    # Enregistrer le DataFrame résultant en tant que fichier CSV
    donnees_concatenees.to_csv(fichier_sortie, index=False)

# Exemple d'utilisation de la fonction
dossier = "centroid_data_3"
fichier_sortie = "fichier_final.csv"
concatener_csv_dossier(dossier, fichier_sortie)