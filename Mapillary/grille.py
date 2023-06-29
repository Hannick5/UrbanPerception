import geopandas as gpd
from shapely.geometry import Polygon

# Coordonnées géographiques des limites de la ville d'Ottawa
ottawa_boundaries = {
    'latitude_min': 44.9218,
    'latitude_max': 45.5877,
    'longitude_min': -76.3655,
    'longitude_max': -75.1731
}

# Taille du carré de grille (en degrés)
grid_size = 0.01

# Calcul du nombre de carrés de grille dans chaque direction
num_rows = int((ottawa_boundaries['latitude_max'] - ottawa_boundaries['latitude_min']) / grid_size)
num_cols = int((ottawa_boundaries['longitude_max'] - ottawa_boundaries['longitude_min']) / grid_size)

# Liste des polygones de la grille
polygons = []

# Création des polygones de la grille
for row in range(num_rows):
    for col in range(num_cols):
        # Calcul des coordonnées géographiques des coins du carré
        min_lat = ottawa_boundaries['latitude_min'] + row * grid_size
        max_lat = ottawa_boundaries['latitude_min'] + (row + 1) * grid_size
        min_lon = ottawa_boundaries['longitude_min'] + col * grid_size
        max_lon = ottawa_boundaries['longitude_min'] + (col + 1) * grid_size
        
        # Création du polygone du carré
        polygon = Polygon([(min_lon, min_lat), (max_lon, min_lat), (max_lon, max_lat), (min_lon, max_lat)])
        polygons.append(polygon)

# Création de la géodataframe à partir des polygones
grid_gdf = gpd.GeoDataFrame({'geometry': polygons})

# Sauvegarde de la géodataframe au format Shapefile
grid_gdf.to_file('data/grid6.shp', driver='ESRI Shapefile')
