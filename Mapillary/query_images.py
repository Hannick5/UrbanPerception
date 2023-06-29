import os
import csv
import requests

def telecharger_images_mapillary(csv_file, output_folder, access_token):
    # Mapillary API endpoint
    url = 'https://graph.mapillary.com/{}?fields=thumb_2048_url'

    # Créer le dossier de stockage s'il n'existe pas déjà
    os.makedirs(output_folder, exist_ok=True)

    # Ouvrir le fichier CSV
    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            image_id = row['HubName']

            # Vérifier si le fichier existe déjà dans le dossier de sortie
            output_path = os.path.join(output_folder, '{}.jpg'.format(image_id))
            if os.path.exists(output_path):
                print('Image {} déjà présente dans le dossier. Ignorer le téléchargement.'.format(image_id))
                continue

            # Construire l'URL pour récupérer l'URL de l'image
            request_url = url.format(image_id)

            # Envoyer une requête GET pour récupérer l'URL de l'image
            response = requests.get(request_url, headers={'Authorization': 'OAuth {}'.format(access_token)})

            # Vérifier si la requête a réussi
            if response.status_code == 200:
                data = response.json()
                image_url = data['thumb_2048_url']

                # Sauvegarder l'image avec l'ID comme nom de fichier dans le dossier de sortie
                with open(output_path, 'wb') as image_file:
                    image_data = requests.get(image_url, stream=True).content
                    image_file.write(image_data)
                print('Image {} téléchargée avec succès.'.format(image_id))
            else:
                print('Échec de récupération de l\'URL de l\'image pour {}.'.format(image_id))

# Utilisation de la fonction
csv_file = 'mapillary_correct_streetview.csv'
output_folder = 'Mapillary/mapillary_training'
access_token = 'MLY|6185142121541114|b499ccd56295db0bff9704d2408e1b90'

telecharger_images_mapillary(csv_file, output_folder, access_token)
