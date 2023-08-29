import os
import rasterio
from PIL import Image

# Rutas de entrada y salida
input_folder = os.path.expanduser('~/datasets/AI4Boundaries2/sentinel2/masks/LU')
output_folder = os.path.expanduser('~/ResUnet-a/masktif')

# Crear la carpeta de salida
os.makedirs(output_folder, exist_ok=True)

# Obtener la lista de archivos .tif en la carpeta de entrada
tif_files = [f for f in os.listdir(input_folder) if f.endswith('.tif')]

# Procesar cada archivo .tif
for tif_file in tif_files:
    tif_file_path = os.path.join(input_folder, tif_file)
    
    with rasterio.open(tif_file_path) as src:
        band = src.read(1)
        band_int = (band * 255).astype('uint8')
        image_tif = Image.fromarray(band_int)
        
        # Crear la ruta de salida para el archivo PNG
        png_file_name = os.path.splitext(tif_file)[0] + '.png'
        png_file_path = os.path.join(output_folder, png_file_name)
        
        # Guardar la imagen PNG
        image_tif.save(png_file_path)

print("Conversión completa")
