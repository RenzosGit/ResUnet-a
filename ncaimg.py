import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Ruta LU
folder_path = os.path.expanduser("~/datasets/AI4Boundaries2/sentinel2/images/LU")

# Ruta imágenes
output_folder = os.path.expanduser("~/ResUnet-a/satelimg")

# Crear la carpeta de salida
os.makedirs(output_folder, exist_ok=True)

# Lista para almacenar las rutas de los archivos LU
nc_files = []

# Iterar a través de los archivos en la carpeta y guardar los archivos .nc
for filename in os.listdir(folder_path):
    if filename.endswith(".nc"):
        nc_files.append(os.path.join(folder_path, filename))

# Iterar a través de los archivos .nc
for nc_file in nc_files:
    # Cargar el archivo con xarray
    ds = xr.open_dataset(nc_file)
    
    # Selecciona las variables de los canales rojo, verde y azul
    red_channel = ds['B4']
    green_channel = ds['B3']
    blue_channel = ds['B2']

    # Crea la imagen RGB
    rgb_image = xr.concat([red_channel, green_channel, blue_channel], dim='band')

    # Transpone las dimensiones para que coincidan con el orden: ('time', 'y', 'x', 'band')
    rgb_image = rgb_image.transpose('time', 'y', 'x', 'band')

    # Remplazar el valor -9999 con NaN para que no afecte la normalización
    red_channel.values[red_channel.values == -9999] = np.nan
    green_channel.values[green_channel.values == -9999] = np.nan
    blue_channel.values[blue_channel.values == -9999] = np.nan

    # Calcular los valores mínimos y máximos (sin tener en cuenta los NaN)
    min_value = np.nanmin([red_channel.values, green_channel.values, blue_channel.values])
    max_value = np.nanmax([red_channel.values, green_channel.values, blue_channel.values])

    # Normaliza los valores entre 0 y 1
    normalized_red = (red_channel.values - min_value) / (max_value - min_value)
    normalized_green = (green_channel.values - min_value) / (max_value - min_value)
    normalized_blue = (blue_channel.values - min_value) / (max_value - min_value)

    # Crea la imagen RGB
    rgb_image_normalized = np.stack((normalized_red, normalized_green, normalized_blue), axis=-1)

    # Asegurar que los valores estén en el rango [0, 1]
    rgb_image_normalized = np.clip(rgb_image_normalized, 0, 1)

    # Guarda la imagen en un archivo
    output_filename = os.path.join(output_folder, os.path.splitext(os.path.basename(nc_file))[0] + ".png")
    plt.imsave(output_filename, rgb_image_normalized[0])
