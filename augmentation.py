from argparse import ArgumentParser
import cv2, os, time
import numpy as np
import random
import multiprocessing as mp
from multiprocessing import Pool

def createImages(input_folder: str, file: str, output_folder: str, number_of_images: int, output_size: tuple):
    """
    Lleva a cabo la creación de imágenes a partir de una realizando operaciones de transformación (rotación)

    Parámetros
    ----------
    input_folder : str
        Carpeta contenedora de la imágen

    file : str
        Nombre del archivo orriginal con el que trabajará

    output_folder : str
        Nombre de la carpeta donde se guardarán las imágenes resultantes

    number_of_images : int
        Cantidad de imágenes a generar a partir de la original
    -------
    
    Returns
    -------
    list
        imágenes generadas por el algoritmo

    """
    # Abriendo la imagen para trabajar
    img = cv2.imread("{}/{}".format(input_folder, file))
    # Conociendo sus dimensiones
    h, w, _ = img.shape

    # Calculando los angulos de acuerdo a la cantidad de imágenes
    angle = int(80/(number_of_images))
    # output_images = []
    print("Trabajando con: ", file)
    for i in range(number_of_images):
        # print("\tGenerating {} of {}\n".format(i+1, number_of_images))
        rotation_angle = angle * (i + 1)
        if i+1 > number_of_images/2:
            rotation_angle = angle * (i - number_of_images)

        # print("\tAngle: {}°\n\tRotation angle: {}°\n".format(angle,rotation_angle))
        
        scale = random.uniform(0.5, 0.9)
        
        # Generando la matriz de rotación
        rotation_matrix = cv2.getRotationMatrix2D((w/2,h/2), rotation_angle, scale)
        new_img = cv2.warpAffine(img, rotation_matrix, (w,h))

        #Si el 

        # Asegurando el tamaño de salida de la imagen
        new_img = cv2.resize(new_img, output_size)

        output_file = "{}/{}{}".format(output_folder, i+1, file)

        # output_images.append(new_img)
        cv2.imwrite(output_file, new_img)
    img = cv2.resize(img, output_size)
    cv2.imwrite("{}/{}".format(output_folder, file), img)
    cv2.destroyAllWindows()


#Definiendo los argumentos para la ejecución del script
argument_parser = ArgumentParser()
argument_parser.add_argument('-i', '--input_folder', 
    required=1, help='Ruta contenedora de las imágenes.')

argument_parser.add_argument('-o', '--output_folder', default="augmentation",
    help='Nombre de carpeta de salida (será creada dentro de la carpeta contenedora).')

argument_parser.add_argument('-ni', '--number_of_images', required=1, 
        type=int,help='Cantidad de imágenes a generar por cada imagen dentro de la carpeta.')
#Definiendo el argumento del modo de procesamiento 
#sc = Single Core (Default)
#mc = Multi-Core
argument_parser.add_argument('-m', '--method', default="sc",
        help='Modo de procesamiento \nsc = Single Core (Default)\nmc = Multi-Core.')

argument_parser.add_argument('-p', '--processes', default=-1, 
        type=int,help='Cantidad de procesos a utilizar.')

argument_parser.add_argument('-os', '--output_size', required=1, 
    type=str,help='Valor en pixeles que tendrán las imágenes de salida, deberá estar en formato (ancho,alto).')



if __name__ == "__main__":
    arguments = vars(argument_parser.parse_args())

    #Dirección de la carpeta contenedora de las imágenes
    input_folder = str(arguments["input_folder"])

    # Número de imágenes a generar
    number_of_images = int(arguments["number_of_images"])

    # Valor en pixeles que tendrán las imágenes de salida
    output_size = str(arguments["output_size"])
    output_size = output_size.strip(' ()')
    print("OS: ", output_size)
    if output_size != "":
        output_size = output_size.split(',')
        num1 = int(output_size[0])
        num2 = int(output_size[1])
        output_size = tuple([num1, num2])
    else:
        print("usage: extraccion_caracteristicas.py [-h] -i INPUT_FOLDER [-o OUTPUT_FOLDER] -os OUTPUT_SIZE [-m METHOD] [-p PROCESSES]\nextraccion_caracteristicas.py: error: the following arguments was wrong writen: -os/--output_size")
        quit()

    #Obteniendo los nombres de archivos dentro de la carpeta
    nombre_archivos = list(os.listdir(input_folder))

    archivos = []
    for archivo in nombre_archivos:
        if archivo.split(".")[-1].lower() in ["jpg", "png", "jpeg", "webp"]:
            archivos.append(archivo)

    nombre_archivos = archivos
    
    #Creación de la carpeta de salida
    output_folder_path =  input_folder + "/" + arguments["output_folder"]
    if(not os.path.exists(output_folder_path)):
        os.mkdir(output_folder_path)
        print("Directorio creado")
    

    inicio = time.time()

    # Revisando el método seleccionado para la ejecución
    if arguments["method"] == "sc": # Usando Solamente el nucleo principal
        results = []
        for archivo in nombre_archivos:

            createImages(input_folder, archivo, output_folder_path, number_of_images, output_size   )
            # if img is None:
            #     continue
            # cv2.imwrite(output_file,img)
    else: 
        if arguments["method"] == "mc": # Trabajando con todos los nucleos del procesador
            procesos = arguments["processes"]
            if procesos < 1:
                procesos = mp.cpu_count()
            results = []
            with Pool(processes=procesos) as pool:
                

                pool.starmap(createImages, 
                    [(input_folder, 
                      archivo, 
                      output_folder_path, 
                      number_of_images,
                      output_size) 
                      for archivo in nombre_archivos])

                pool.close()

            # with Pool(processes=procesos) as pool:
            #     pool.starmap(saveFile, [(output_file, img) for output_file, img in results])
            #     pool.close()

            # for output_file,img in results:
            #     if img is None:
            #         continue
            #     cv2.imwrite(output_file,img)
        else : # Opción inexistente
            print("El método de procesamiento no es una opción válida")
            quit()

    fin = time.time()

    # print ("\n\nProceso finalizado en {} milisegundo".format(fin-inicio))
    print ("\n\nProceso finalizado en %0.5f segundos" % (fin-inicio))