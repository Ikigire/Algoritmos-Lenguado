import math
import cv2
import easyocr
from OperacionesIMG import reduccionEscalaGrises


def conteoPigmentosBlancos(img: math)-> int:
    """
    Función para realizar en conteo de pigmentos BLANCOS dentro de la imagen de un dorsal de pez

    Parámetros
    ----------
    img : math
        Imagen a la que se le hará el conteo de pigmentos blancos, esta imagen tiene que estar en formato BGR
    -------
    
    Returns
    -------
    int
        Cantidad de pigmentos blancos encontrados denro de la imagen
    """
    #reducción de ruido
    img = cv2.GaussianBlur(img, (3,3), 2)

    #cambio de formato de color
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #uniformizando la escala de grises
    img = reduccionEscalaGrises(img, 35, 65, 90)

    # Aplicando binarización para segmentación de imágenes
        # Threshold por valor
    _, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Conteo de pigmentos Blancos
    pigmentos_blancos = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    pigmentos_blancos = list(filter(lambda contour: cv2.contourArea(contour) > 45, pigmentos_blancos))

    return len(pigmentos_blancos)

def conteoPigmentosNegros(img: math)-> int:
    """
    Función para realizar en conteo de pigmentos NEGROS dentro de la imagen de un dorsal de pez

    Parámetros
    ----------
    img : math
        Imagen a la que se le hará el conteo de pigmentos negros, esta imagen tiene que estar en formato BGR
    -------
    
    Returns
    -------
    int
        Cantidad de pigmentos negros encontrados denro de la imagen
    """
    #reducción de ruido
    img = cv2.GaussianBlur(img, (3,3), 2)

    #cambio de formato de color
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #uniformizando la escala de grises
    # img = reduccionEscalaGrises(img, 35, 65, 90)

    # Aplicando binarización para segmentación de imágenes
        # Threshold Adaptativo
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 271, 7)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 55, 7)


    # Conteo de pigmentos Negros
    pigmentos_negros = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    pigmentos_negros = list(filter(lambda contour: cv2.contourArea(contour) > 45, pigmentos_negros))

    return len(pigmentos_negros)

def getEscala(img: math)-> tuple(int, int, str):
    """
    Realiza la sustracción de la escala, indicando la longitud en pixeles, el valor numérico y la unidad de medida dentro de la imagen recibida

    Parámetros
    ----------
    img : math
        Imagen que será tratada por el método, esta deberá estar en forma BGR
    -------
    
    Returns
    -------
    int
        Cantidad de pixeles de la escala

    int | NONE
        valor numérico encontrado dentro del texto de la escala, NONE si no se identifica

    str
        texto que pretende indicar la unidad de medida de la escala
    """
    #Convertir a escala de grises
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Generar imagen binaria
    # _, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, thresh = cv2.threshold(gray_img, 105, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRUNC)


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
    escala, unidad_medida = getTextoEscala(gray_img_text)
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
    _,_,longitud_escala,_ = cv2.boundingRect(img_contours[0])

    return longitud_escala, escala, unidad_medida



def getTextoEscala(image: math)-> tuple(int, str):
    """
    Función que realiza la lectura de texto en una imagen en busca de escala de forma que 
    se extraiga un valor numérico y texto

    Parámetros
    ----------
    img : math
        Imagen que contiene el texto a ser interpetado,
        esta imagen debe de estar en escala de grises para funcionar
    -------
    
    Returns
    -------
    int | NONE
        Valor numérico del texto de la escala, NONE si no fue posible identificar un valor numérico
    
    str
        Valor de texto que acompaña el valor numérico
    """
    escala = ""
    unidad_medida = ""
    texto_escala = None

    intentos = 3
    reintentar = True

    # Reescalando la imagen para poder realizar la lectura del texto sin mayor complicación
    size = list(image.shape)
    size[0] *= 4
    size[1] *= 4

    image = cv2.resize(image, tuple(reversed(size)), interpolation= cv2.INTER_AREA)

    while reintentar and intentos>=0:
        reintentar = False
        try:
            # Realizando la detección del texto de la escala
            reader = easyocr.Reader(['es'], gpu=False)
            textos = reader.readtext(image)

            
            texto_escala = textos[0][1]

            for char in texto_escala:
                if char in "0123456789":
                    escala += char
                else:
                    unidad_medida += char

            escala = int(escala)
            unidad_medida = unidad_medida.strip()
        except :
            print("Intentando detectar escala nuevamente")
            reintentar = True
            intentos -=1
            if intentos <0:
                raise Exception("No fue posible determinar el texto de la escala: Error de lectura de escala")
    
    if reintentar:
        escala = None

    return escala, unidad_medida

def segmentacionPez(image: math)-> math:
    """
    Método que lleva a cabo una reducción en los niveles de escala de grises para lograr destacar objetos sobre la imagen

    Parámetros
    ----------
    img : math
        Imagen que será tratada por el método

    umbral_scale_1 : int
        Valor de umbral número 1 para la reducción de escala
    
    umbral_scale_2 : int
        Valor de umbral número 1 para la reducción de escala

    umbral_scale_3 : int
        Valor de umbral número 1 para la reducción de escala
    -------
    
    Returns
    -------
    math
        Imagen modificada con la reducción de escala apliacada
    """
    #Recortar de la imágen para eliminar el pez
    img = image[405::,0:615]

    #Convertir a escala de grises
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #Aplicando algunos filtros para mejorar la detección de bordes
    gray_img = cv2.GaussianBlur(gray_img, (3,3), 2)
    
    #Generar imagen binaria
    _, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return []