import cv2, easyocr, time, os
from argparse import ArgumentParser
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import pandas as pd

from OperacionesIMG import reduccionEscalaGrises

# "ID_pez", "peso", "Long", "ASDA", "IEDA", "ASFA", "IEFA", "FAC", "TAC", "FANE", "TANE", "glucosa_in", "lactosa_in", "cortisol_in", "glucosa_est", "lactosa_est", "cortisol_est", "opercula", "ojo_ojo", "ojo", "boca", "agalla", "aleta", "nariz", "ojo_cachete", "poro_aleta", "ancho", "ojo_agalla", "agalla_nariz", "agalla_boca"

def extraer_caracteristicas_carpeta(input_folder: str, output_folder_name: str = "output", output_size: tuple = (800,600), method: str = "sc", processes: int = 0, save_ouput: bool = True):
    if input_folder is None:
        raise Exception("input_folder field is required")
    #Obteniendo los nombres de archivos dentro de la carpeta
    nombre_archivos = os.listdir(input_folder)
    print("\n\n\n\n\nNombre de los archivos: {}\n\n\n\n\n".format(nombre_archivos))
    results = []
    
    #Creación de la carpeta de salida
    output_folder_path =  input_folder + "/" + output_folder_name
    if(not os.path.exists(output_folder_path)) and save_ouput:
        os.mkdir(output_folder_path)
        print("Directorio creado")

    # Revisando el método seleccionado para la ejecución
    if method == "sc": # Usando Solamente el nucleo principal
        for archivo in nombre_archivos:

            results.append(extraer_caracteristicas(input_folder, archivo, output_folder_path, output_size))
            # if img is None:
            #     continue
            # cv2.imwrite(output_file,img)

        if save_ouput:
            for result in results:
                pez, dorsal = result
                if not pez is None:
                    saveFile(pez[1],pez[0])
                    saveFile(dorsal[1],dorsal[0])
    else: 
        if method == "mc": # Trabajando con todos los nucleos del procesador
            processes
            if processes < 1:
                processes = mp.cpu_count()

            with Pool(processes=processes) as pool:
                

                results = pool.starmap(extraer_caracteristicas, 
                    [(input_folder, 
                      archivo, 
                      output_folder_path, 
                      output_size) 
                      for archivo in nombre_archivos])

                pool.close()
            
            if save_ouput:
                for result in results:
                    # print("\t\t",result)
                    pez, dorsal = result
                    if not pez is None:
                        saveFile(pez[1],pez[0])
                        saveFile(dorsal[1],dorsal[0])

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

    return results

def extraer_caracteristicas(folder_path, archivo, output_path, output_size, remove_background):
    """Función de busca extraer las características de una imagen que contiene un pez

    Parámetros
    ----------
    folder_path : string
        Ruta de la carpeta contenedora en donde se encuentra la imagen de entrada

    archivo : string
        Nombre del archivo con el cual se trabajará (También define el nombre de la imagen de salida)
    
    output_path : string
        Ruta de la carpeta contenedora de la salida (Imagen resultante)
    
    output_size : float
        Proporción de la escala para la salida
    -------
    
    Returns
    -------
    tuple
        Array con la diversa información resultado del preprocesamiento y segmentación de la imagen
    """

    if (archivo.lower().split(".")[-1] not in ["jpeg", "png", "jpg"]):
        return None, None

    image_path = folder_path + "/" + archivo
    
    image = cv2.imread(image_path)

    if image is None:
        return None

    #""###########################################################################""#
    #""|                                      	|                                   #
    #""|                                      	|                 /|                #
    #""|  Detección de la escala en pixeles   	|                / |                #
    #""|		Y lectura del texto				|                  |                #
    #""|										|               ___|___             #
    #""|										|                                   #
    #""###########################################################################""#

    #Recortar de la imágen para eliminar el pez
    img = image[405::,0:615]

    # print("\t\n\n\nARCHIVO: {}\n\n\n".format(archivo))

    #Convertir a escala de grises
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    reduccionEscalaGrises(gray_img)

    #Generar imagen binaria
    _, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #Detección de contornos
    #img_contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    img_contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    img_contours = sorted(img_contours, key=cv2.contourArea)
    contorno = img_contours[0]
    if len(img_contours) > 1:
        length = img.shape[2]
        menor = length
        for c in img_contours:
            x,y,w,h = cv2.boundingRect(c)
            nuevo_valor = length-x
            if nuevo_valor < menor and w > 60 and h > 10:
                contorno = c
                menor = nuevo_valor
    #Volviendo a recortar la imágen para tratar de omitir la región en blanco y el tecto que tiene la escala

    #Con esta linea logras obtener los atributos de la longitud
    #   x=pixel donde empieza en el eje horizontal, 
    #   y=pixel donde empieza en el eje vertical,
    #   w=ancho en pixeles de la imagen,
    #   h=alto en pixeles de la imagen
    x,y,w,h = cv2.boundingRect(contorno)
    
    # Realizando una copia de la imágen para leer el texto de la escala
    gray_img_text = gray_img[y+4: y+h, x: x+w]
    
    # Recortando la imágen para obtener únicamente la escala
    gray_img = gray_img[y+2:int(y+h/2)-2,x+2:x+w-2]

    # Volvemos a repetir el proceso completo pero ahora la finalidad es obtener únicamente la linea que representa la escala
    # Generar imagen binaria
    _, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)

    #Detección de contornos ahora para la detección única de la escala
    #img_contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    img_contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    img_contours = sorted(img_contours, key=cv2.contourArea)

    #Con esta linea logras obtener los atributos de la longitud
    x,y,longitud_escala,h = cv2.boundingRect(img_contours[0])
    
    # Reescalando la imagen para poder realizar la lectura del texto sin mayor complicación
    size = list(gray_img_text.shape)
    size[0] *= 4
    size[1] *= 4

    gray_img_text = cv2.resize(gray_img_text, tuple(reversed(size)), interpolation= cv2.INTER_AREA)

    escala = ""
    unidad_medida = ""
    texto_escala = None

    intentos = 3
    reintentar = True

    while reintentar and intentos>=0:
        reintentar = False
        try:
            # Realizando la detección del texto de la escala
            reader = easyocr.Reader(['es'], gpu=False)
            textos = reader.readtext(gray_img_text)

            
            texto_escala = textos[0][1]
            # print("\t\t\t\n\n\nTEXTO: {}\n\n\n".format(texto_escala))

            for char in texto_escala:
                if char in "0123456789":
                    escala += char
                else:
                    unidad_medida += char

            # print("\t\t\t\n\n\nESCALA: {}\n\n\n".format(escala))
            escala = int(escala)
        except :
            print("Intentando detectar escala nuevamente")
            reintentar = True
            intentos -=1
            if intentos <0:
                exit("No fue posible terminar el trabajo: Error de lectura de escala")

    
    unidad_medida = unidad_medida.strip()
    proporcion = longitud_escala/escala

    #""###########################################################################""#
    #""|                                      	|            ______                 #
    #""|                                      	|           |      |                #
    #""|  Detección de bordes y segmentación   	|                  |                #
    #""|										|            ______|                #
    #""|                                        |           |                       #
    #""|										|           |_______                #
    #""###########################################################################""#

    #Recortar una imagen
    img = image[0:445,0::]

    #Convertir a escala de grises
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (3,3), 2)
    
    #Generar imagen binaria
    _, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    #Detección de contornos
    img_contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    img_contours = sorted(img_contours, key=cv2.contourArea)

    i = img_contours[-1]

    #Generar mascara
    mask = np.zeros(img.shape[:2], np.uint8)
    #Dibujar contornos
    cv2.drawContours(mask, img_contours, len(img_contours)-1, 255, -1)
    new_img = None
    # print("Remove bg: ",remove_background)
    if remove_background:
        #Substracción del fondo
        new_img = cv2.bitwise_and(img, img, mask=mask)
    else:
        new_img = img.copy()
    

    #Genera la segunda segmentación
    x, y, w, h = cv2.boundingRect(i)
    tercia_h = int(w/3)
    tercia_v = int(h/3)

    os_w, os_h = output_size

    longitud_pez = os_w*100/w
    altura_pez   = os_h*100/h

    dorsal = new_img[y+tercia_v:y+h-tercia_v, x+tercia_h:x+w-tercia_h]

    size = list(dorsal.shape[:2])
    
    size[0] = int(size[0] * 3)
    size[1] = int(size[1] * 3)
    
    longitud_dorsal = size[1]
    altitud_dorsal  = size[0]
    
    dorsal = cv2.resize(dorsal, tuple(reversed(size)), interpolation= cv2.INTER_AREA)

    # Ajustando la imagen final
    new_img = new_img[y-30: y+h+30, x: x+w]

    size = list(new_img.shape[:2])
    
    # size[0] *= os
    # size[1] *= os

    new_img = cv2.resize(new_img, output_size, interpolation= cv2.INTER_AREA)

    #reducción de ruido
    dorsal_filtrado = cv2.GaussianBlur(dorsal, (3,3), 2)
    
    dorsal_filtrado = cv2.cvtColor(dorsal_filtrado, cv2.COLOR_BGR2GRAY)

    dorsal_filtrado2 = dorsal_filtrado.copy()

    
    #Reduciendo la escala de grises
    # dorsal_filtrado = reduccionEscalaGrises(dorsal_filtrado, 35, 65, 90)
    # dorsal_filtrado2 = reduccionEscalaGrises(dorsal_filtrado2, 100,155,215)
    

    # dorsal_filtrado = cv2.adaptiveThreshold(dorsal_filtrado, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 9)
    # dorsal_filtrado = cv2.adaptiveThreshold(dorsal_filtrado, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 271, 7)
    dorsal_filtrado = cv2.adaptiveThreshold(dorsal_filtrado, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 55, 7)
    # dorsal_filtrado2 = cv2.adaptiveThreshold(dorsal_filtrado2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY_INV, 601, 31)
    # _, dorsal_filtrado2 = cv2.threshold(dorsal_filtrado2, 130, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, dorsal_filtrado2 = cv2.threshold(dorsal_filtrado2, 105, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRUNC)


    # dorsal_filtrado = cv2.adaptiveThreshold(dorsal_filtrado, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 71, 13)

    # Conteo de pigmentos Negros
    pigmentos_negros = cv2.findContours(dorsal_filtrado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    pigmentos_negros = list(filter(lambda contour: cv2.contourArea(contour) > 45, pigmentos_negros))

    # Conteo de pigmentos Blancos
    pigmentos_blancos = cv2.findContours(dorsal_filtrado2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    pigmentos_blancos = list(filter(lambda contour: cv2.contourArea(contour) > 45, pigmentos_blancos))

    # print("\n\nPigmentos Negros: {}\nPigmentos Blancos: {}\n\n".format(len(pigmentos_negros), len(pigmentos_blancos)))
    #Calculando las medidas del pez
    longitud_pez = longitud_pez/proporcion
    altura_pez   = altura_pez/proporcion

    longitud_dorsal = longitud_dorsal/(proporcion*3)
    altitud_dorsal  = altitud_dorsal/(proporcion*3)

    # Guardando datos del pez
    datos_pez = []
    datos_dorsal = []


    #Datos del Pez
    # datos_pez.append(new_img)
    # datos_pez.append("{}/{}{}".format(output_path, "pez/new_", archivo))
    # datos_pez.append(longitud_pez)
    # datos_pez.append(altura_pez)
    # datos_pez.append(unidad_medida)
    #Datos del Dorsal
    # datos_dorsal.append(dorsal)
    # datos_dorsal.append("{}/{}{}".format(output_path, "dorsal/new_dorsal_", archivo))
    # datos_dorsal.append(longitud_dorsal)
    # datos_dorsal.append(altitud_dorsal)
    # datos_dorsal.append((longitud_dorsal * altitud_dorsal))
    # datos_dorsal.append(len(pigmentos_negros))
    # datos_dorsal.append((len(pigmentos_negros)/(longitud_dorsal*altitud_dorsal)))
    # datos_dorsal.append(len(pigmentos_blancos))
    # datos_dorsal.append((len(pigmentos_blancos)/(longitud_dorsal*altitud_dorsal)))
    # datos_dorsal.append(unidad_medida)
    datos_pez = [new_img, "{}/{}{}".format(output_path, "pez/new_", archivo), longitud_pez, altura_pez, unidad_medida]
    datos_dorsal = [dorsal, "{}/{}{}".format(output_path, "dorsal/new_dorsal_", archivo), longitud_dorsal, altitud_dorsal, (longitud_dorsal*altitud_dorsal), len(pigmentos_negros), (len(pigmentos_negros)/(longitud_dorsal*altitud_dorsal)), len(pigmentos_blancos), (len(pigmentos_blancos)/(longitud_dorsal*altitud_dorsal)), unidad_medida]
    
    return [datos_pez, datos_dorsal]


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

argument_parser.add_argument('-os', '--output_size', required=1, 
    type=str,help='Valor en pixeles que tendrán las imágenes de salida, deberá estar en formato (ancho,alto).')
#Definiendo el argumento del modo de procesamiento 
#sc = Single Core (Default)
#mc = Multi-Core
argument_parser.add_argument('-m', '--method', default="sc",
        help='Modo de procesamiento \nsc = Single Core (Default)\nmc = Multi-Core.')
argument_parser.add_argument('-p', '--processes', default=-1, 
        type=int,help='Cantidad de procesos a utilizar.')
argument_parser.add_argument('-ss', '--skip_save', default=0, 
        type=bool,help='Omitir el guardado de las imágenes.')
argument_parser.add_argument('-rb', '--remove_background', default=1, 
        type=int,help='Omitir el guardado de las imágenes.')



if __name__ == "__main__":
    arguments = vars(argument_parser.parse_args())

    #Dirección de la carpeta contenedora de las imágenes
    input_folder = arguments["input_folder"]

    #Definiendo si se guardan las imágenes
    skip_save = bool(arguments["skip_save"])

    #Definiendo si se elimina el fondo o no
    remove_background = bool(int(arguments["remove_background"]))
    # print(arguments["remove_background"])
    # print("Remove Background: ", remove_background)

    #Obteniendo los nombres de archivos dentro de la carpeta
    nombre_archivos = list(os.listdir(input_folder))

    archivos = []
    for archivo in nombre_archivos:
        if archivo.split(".")[-1].lower() in ["jpg", "png", "jpeg"]:
            archivos.append(archivo)
    nombre_archivos = archivos
    
    #Creación de la carpeta de salida
    output_folder_path =  input_folder + "/" + arguments["output_folder"]
    if(not os.path.exists(output_folder_path)):
        os.mkdir(output_folder_path)
        os.mkdir(output_folder_path + "/pez")
        os.mkdir(output_folder_path + "/dorsal")
        # print("Directorio creado")

    # Valor en pixeles que tendrán las imágenes de salida
    output_size = str(arguments["output_size"])
    output_size = output_size.strip(' ()')
    # print("OS: ", output_size)
    if output_size != "":
        output_size = output_size.split(',')
        num1 = int(output_size[0])
        num2 = int(output_size[1])
        output_size = tuple([num1, num2])
    else:
        print("usage: extraccion_caracteristicas.py [-h] -i INPUT_FOLDER [-o OUTPUT_FOLDER] -os OUTPUT_SIZE [-m METHOD] [-p PROCESSES]\nextraccion_caracteristicas.py: error: the following arguments was wrong writen: -os/--output_size")
        quit()
    

    inicio = time.time()

    # Revisando el método seleccionado para la ejecución
    if arguments["method"] == "sc": # Usando Solamente el nucleo principal
        results = []
        for archivo in nombre_archivos:

            results.append(extraer_caracteristicas(input_folder, archivo, output_folder_path, output_size, remove_background))
            # if img is None:
            #     continue
            # cv2.imwrite(output_file,img)

        df = pd.DataFrame()
        if not skip_save:
            imagenes_dorsales = []
            imagenes_peces = []
            print("Resultados:")
            for result in results:
                pez, dorsal = result
                if not pez is None:
                    saveFile(pez[1],pez[0])
                    imagenes_peces.append(pez[1])
                    saveFile(dorsal[1],dorsal[0])
                    imagenes_dorsales.append(dorsal[1])
            
            df['peces'] = imagenes_peces
            df['dorsales'] = imagenes_dorsales

        # from pez
        longitudes = []
        altitudes = []
        
        # From dorsal
        longitudes_dorsal = []
        altitudes_dorsal = []
        area_dorsal = []
        pigmentos_negros = []
        dispersion_pigmentos_negros = []
        pigmentos_blancos = []
        dispersion_pigmentos_blancos = []
        for result in results:
            pez, dorsal = result
            longitudes.append(pez[2])
            altitudes.append(pez[3])
            
            longitudes_dorsal.append(dorsal[2])
            altitudes_dorsal.append(dorsal[3])
            area_dorsal.append(dorsal[4])
            pigmentos_negros.append(dorsal[5])
            dispersion_pigmentos_negros.append(dorsal[6])
            pigmentos_blancos.append(dorsal[7])
            dispersion_pigmentos_blancos.append(dorsal[8])
        
        df['longitud'] = longitudes
        df['altura'] = altitudes
        df['longitud_dorsal'] = longitudes_dorsal
        df['altitud_dorsal'] = altitudes_dorsal
        df['area_dorsal'] = area_dorsal
        df['pig_negros'] = pigmentos_negros
        df['disp_pig_negros'] = dispersion_pigmentos_negros
        df['pig_blancos'] = pigmentos_blancos
        df['disp_pig_blancos'] = dispersion_pigmentos_blancos

        df.to_csv("{}/{}".format(output_folder_path, "datos_algoritmo.csv"), columns=df.columns)
    else: 
        if arguments["method"] == "mc": # Trabajando con todos los nucleos del procesador
            procesos = arguments["processes"]
            if procesos < 1:
                procesos = mp.cpu_count()
            results = []
            with Pool(processes=procesos) as pool:
                

                results = pool.starmap(extraer_caracteristicas, 
                    [(input_folder, 
                      archivo, 
                      output_folder_path, 
                      output_size,
                      remove_background) 
                      for archivo in nombre_archivos])

                pool.close()
            df = pd.DataFrame()
            if not skip_save:
                imagenes_dorsales = []
                imagenes_peces = []
                print("Resultados:")
                for result in results:
                    pez, dorsal = result
                    if not pez is None:
                        saveFile(pez[1],pez[0])
                        imagenes_peces.append(pez[1])
                        saveFile(dorsal[1],dorsal[0])
                        imagenes_dorsales.append(dorsal[1])
                
                df['peces'] = imagenes_peces
                df['dorsales'] = imagenes_dorsales
            # From pez
            longitudes = []
            altitudes = []

            # From dorsal
            longitudes_dorsal = []
            altitudes_dorsal = []
            area_dorsal = []
            pigmentos_negros = []
            dispersion_pigmentos_negros = []
            pigmentos_blancos = []
            dispersion_pigmentos_blancos = []
            for result in results:
                pez, dorsal = result
                longitudes.append(pez[2])
                altitudes.append(pez[3])
                
                longitudes_dorsal.append(dorsal[2])
                altitudes_dorsal.append(dorsal[3])
                area_dorsal.append(dorsal[4])
                pigmentos_negros.append(dorsal[5])
                dispersion_pigmentos_negros.append(dorsal[6])
                pigmentos_blancos.append(dorsal[7])
                dispersion_pigmentos_blancos.append(dorsal[8])
            
            df['longitud'] = longitudes
            df['altura'] = altitudes
            df['longitud_dorsal'] = longitudes_dorsal
            df['altitud_dorsal'] = altitudes_dorsal
            df['area_dorsal'] = area_dorsal
            df['pig_negros'] = pigmentos_negros
            df['disp_pig_negros'] = dispersion_pigmentos_negros
            df['pig_blancos'] = pigmentos_blancos
            df['disp_pig_blancos'] = dispersion_pigmentos_blancos

            df.to_csv("{}/{}".format(output_folder_path, "datos_algoritmo.csv"), columns=df.columns)

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
