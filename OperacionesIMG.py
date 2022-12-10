import math
import cv2


def reduccionEscalaGrises(img:math, umbral_scale_1:int = 70, umbral_scale_2:int = 130, umbral_scale_3:int = 180) -> math:
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
    # img = cv2.imread("fondo.png",0)

    height,width = img.shape
    for x in range(width):
        for y in range(height):
            if img[y,x] < umbral_scale_1:
                img[y,x] = 0
            elif img[y,x] < umbral_scale_2:
                img[y,x] = 85
            elif img[y,x] < umbral_scale_3:
                img[y,x] = 170
            else:
                img[y,x] = 255
    
    return img


def reduccionEscalaGrisesInvertido(img:math, umbral_scale_1:int = 70, umbral_scale_2:int = 130, umbral_scale_3:int = 180) -> math:
    """"""
    # img = cv2.imread("fondo.png",0)

    h,w = img.shape

    cv2.imshow("Original", img)
    cv2.waitKey(1000)
    for x in range(w):
        cv2.imshow("Resultado", img)
        cv2.waitKey(10)
        for y in range(h):
            if img[y,x] < umbral_scale_1:
                img[y,x] = 255
            else:
                if img[y,x] < umbral_scale_2:
                    img[y,x] = 170
                else:
                    if img[y,x] < umbral_scale_3:
                        img[y,x] = 85
                    else:
                        img[y,x] = 0
    
    return img