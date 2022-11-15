from argparse import ArgumentParser
import cv2, os
import numpy as np

def getListFromImage(input_folder: str, file: str, color: bool) -> list:
    """
    Lleva a cabo la creación de imágenes a partir de una realizando operaciones de transformación (rotación)

    Parámetros
    ----------
    input_folder : str
        Carpeta contenedora de la imágen

    file : str
        Nombre del archivo orriginal con el que trabajará
    -------
    
    Returns
    -------
    list
        imágenes generadas por el algoritmo

    """
    if color:
        img = cv2.imread("{}/{}".format(input_folder, file))
    else: 
        img = cv2.imread("{}/{}".format(input_folder, file), 0)
    return img

#Definiendo los argumentos para la ejecución del script
argument_parser = ArgumentParser()

argument_parser.add_argument('-i', '--input_folder', required=1, help='Ruta contenedora de las imágenes.')

argument_parser.add_argument('-o', '--output_name', default="data", help='Nombre del archivo de salida (será creada dentro de la carpeta contenedora).')

argument_parser.add_argument('-y', '--y_value', required=1, type=int, help='Valor con el que será marcado el conjunto de datos')

argument_parser.add_argument('-c', '--color', required=1, type=str,help='Definir si la salida estará en formato de RGB (1) o en escala de grises (0)')

if __name__ == "__main__":
    
    arguments = vars(argument_parser.parse_args())

    #Dirección de la carpeta contenedora de las imágenes
    input_folder = str(arguments["input_folder"])

    try: 
        in_color = int(arguments["color"])
        in_color = bool(in_color)
    except:
        print("Ocurrio un error:\n\tArgument 'color' parse error")
        quit(0)

    # Obteniendo el valor de etiqueta del conjunto
    y_value = int(arguments["y_value"])

    #Obteniendo los nombres de archivos dentro de la carpeta
    nombre_archivos = list(os.listdir(input_folder))

    archivos = []
    for archivo in nombre_archivos:
        if archivo.split(".")[-1].lower() in ["jpg", "png", "jpeg"]:
            archivos.append(archivo)

    nombre_archivos = archivos

    x_output_name = "{}/{}{}".format(input_folder, "X_",str(arguments["output_name"]) + ".npy")
    y_output_name = "{}/{}{}".format(input_folder,"Y_",str(arguments["output_name"]) + ".npy")

    X = []
    Y = []


    for archivo in archivos:
        X.append(getListFromImage(input_folder, archivo, in_color))
        Y.append([y_value])

    X = np.asanyarray(X)
    Y = np.asanyarray(Y)

    np.save(x_output_name, X)
    np.save(y_output_name, Y)

    in_file_x = np.load(x_output_name)
    in_file_y = np.load(y_output_name)

    print("Final Shape X - Y: ", in_file_x.shape, " - ", in_file_y.shape)