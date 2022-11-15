import os
import time
import cv2
from argparse import ArgumentParser
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool

def escalar(folder_path, archivo, output_path, os):
    """"""
    if (archivo.lower().split(".")[-1] not in ["jpeg", "png", "jpg"]):
        return None, None

    image_path = folder_path + "/" + archivo
    # print("Ruta archivo: " + image_path)
    image = cv2.imread(image_path)

    if image is None:
        return None, None

    #Recortar una imagen
    img = image
    #[0:405,0:615]

    #cv2.imshow('Imagen recortada',img)

    #Convertir a escala de grises
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Generar imagen binaria
    _, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 3)

    #Detección de contornos
    img_contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    img_contours = sorted(img_contours, key=cv2.contourArea, reverse=1)

    for i in img_contours:
        if cv2.contourArea(i) > 100:
            break

    #Genera la segunda segmentación
    x, y, w, h = cv2.boundingRect(i)

    #Generar mascara
    mask = np.zeros(img.shape[:2], np.uint8)
    #Dibujar contornos
    cv2.drawContours(mask, [i],-1, 255, -1)
    #Subtracción del fondo
    new_img =cv2.bitwise_and(img, img, mask=mask)
    
    # Haciendo el resize de la imagen para hacerla cuadrada
    new_img = cv2.resize(new_img, os, interpolation=cv2.INTERSECT_NONE)

    #creando el path de salida del archivo
    output_path = output_path + "/" + archivo
    return output_path, new_img

def saveFile(output_file, img):
    if img is None:
        return
    cv2.imwrite(output_file, img)

#Definiendo los argumentos para la ejecución del script
argument_parser = ArgumentParser()
argument_parser.add_argument('-i', '--input_folder', 
    required=1, help='Ruta contenedora de las imágenes.')
argument_parser.add_argument('-o', '--output_folder', default="output",
    help='Nombre de carpeta de salida (será creada dentro de la carpeta contenedora).')

argument_parser.add_argument('-os', '--output_size', default=-1, 
    type=int,help='Valor en pixeles que tendrán las imágenes de salida.')
#Definiendo el argumento del modo de procesamiento 
#sc = Single Core (Default)
#mc = Multi-Core
argument_parser.add_argument('-m', '--method', default="sc",
        help='Modo de procesamiento \nsc = Single Core (Default)\nmc = Multi-Core.')
argument_parser.add_argument('-p', '--processes', default=-1, 
        type=int,help='Cantidad de procesos a utilizar.')



if __name__ == "__main__":
    arguments = vars(argument_parser.parse_args())

    #Dirección de la carpeta contenedora de las imágenes
    input_folder = arguments["input_folder"]

    #Obteniendo los nombres de archivos dentro de la carpeta
    nombre_archivos = os.listdir(input_folder)

    #Creación de la carpeta de salida
    output_folder_path =  input_folder + "/" + arguments["output_folder"]
    if(not os.path.exists(output_folder_path)):
        os.mkdir(output_folder_path)
        print("Directorio creado")

    # Valor en pixeles que tendrán las imágenes de salida
    output_size = arguments["output_size"]
    if output_size < 1:
        archivo_muestra = input_folder + "/" + nombre_archivos[int((len(nombre_archivos)/2))]
        img = cv2.imread(archivo_muestra)
        h,w,_ = img.shape
        output_size = (h,w)
    else :
        output_size = (output_size, output_size)

    inicio = time.time()

    # Revisando el método seleccionado para la ejecución
    if arguments["method"] == "sc": # Usando Solamente el nucleo principal
        for archivo in nombre_archivos:

            output_file,img = escalar(input_folder, archivo, output_folder_path, output_size)
            if img is None:
                continue
            cv2.imwrite(output_file,img)
    else: 
        if arguments["method"] == "mc": # Trabajando con todos los nucleos del procesador
            procesos = arguments["processes"]
            if procesos < 1:
                procesos = mp.cpu_count()
            results = []
            with Pool(processes=procesos) as pool:
                

                results = pool.starmap(escalar, 
                    [(input_folder, 
                      archivo, 
                      output_folder_path, 
                      output_size) 
                      for archivo in nombre_archivos])

                pool.close()

            with Pool(processes=procesos) as pool:
                pool.starmap(saveFile, [(output_file, img) for output_file, img in results])
                pool.close()

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
