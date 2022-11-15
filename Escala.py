import cv2
import numpy as np

#Leer imagen
image = cv2.imread("pez70.JPG")
# image = cv2.resize(image, (500, 500), interpolation=cv2.INTERSECT_NONE)
#Mostrar imagen
cv2.imshow("Original imagen",image)
cv2.waitKey(0)

#Filas y columnas de la imagen
print('image.shape=',image.shape)

#Recortar de la imágen para eliminar el pez
img = image[405:444,0:615]

#cv2.imshow('Imagen recortada',img)

#Convertir a escala de grises
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Mostrar imagen
cv2.imshow("Gray Scale",gray_img)
cv2.waitKey(0)

#Generar imagen binaria
_, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#Mostrar Imagen
cv2.imshow("Binary Image",thresh)
cv2.waitKey(0)

#Detección de contornos
#img_contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
img_contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
img_contours = sorted(img_contours, key=cv2.contourArea)

#Volviendo a recortar la imágen para tratar de omitir la región en blanco y el tecto que tiene la escala
print("\nContornos encontrados = ", len(img_contours))
if len(img_contours) == 1:
    #Con esta linea logras obtener los atributos de la longitud
    x,y,w,h = cv2.boundingRect(img_contours[0])
    gray_img = gray_img[y+2:int(y+h/2)-2,x+2:x+w-2]

#Volvemos a repetir el proceso completo pero ahora la finalidad es obtener únicamente la linea que representa la escala
cv2.imshow("Gray Scale",gray_img)
cv2.waitKey(0)
# for i in img_contours:
# 	if cv2.contourArea(i) > 100:
# 		break

#Generar imagen binaria
_, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)
#Mostrar Imagen
cv2.imshow("Binary Image",thresh)
cv2.waitKey(0)

#Detección de contornos ahora para la detección única de la escala
#img_contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
img_contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
img_contours = sorted(img_contours, key=cv2.contourArea)
print("Contornos encontrados = ", len(img_contours))

#Con esta linea logras obtener los atributos de la longitud
#   x=pixel donde empieza en el eje horizontal, 
#   y=pixel donde empieza en el eje vertical,
#   w=ancho en pixeles de la imagen,
#   h=alto en pixeles de la imagen
x,y,w,h = cv2.boundingRect(img_contours[0])
print("\n\nLa cantidad de pixeles de ancho de la escala es ", (w))

#Guardar imagen
# cv2.imwrite("pez70_new.jpg",new_img)
