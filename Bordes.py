import cv2
import easyocr
import numpy as np

#Leer imagen
image = cv2.imread("pez70.JPG")
# image = cv2.resize(image, (500, 500), interpolation=cv2.INTERSECT_NONE)
#Mostrar imagen
cv2.imshow("Original imagen",image)
cv2.waitKey(0)

#Filas y columnas de la imagen
print('image.shape=',image.shape)

#Recortar una imagen
img = image[0:405,0:615]

#cv2.imshow('Imagen recortada',img)

#Convertir a escala de grises
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Mostrar imagen
cv2.imshow("Gray Scale",gray_img)
cv2.waitKey(0)

#Generar imagen binaria
_, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 3)
#Mostrar Imagen
cv2.imshow("Binary Image",thresh)
cv2.waitKey(0)

#Detección de contornos
#img_contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
img_contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
img_contours = sorted(img_contours, key=cv2.contourArea)

cv2.drawContours(gray_img, img_contours, -1, 255, 3)
cv2.imshow("Binary Image",thresh)

for i in img_contours:
	if cv2.contourArea(i) > 400:
		break

#Generar mascara
mask = np.zeros(img.shape[:2], np.uint8)
#Dibujar contornos
cv2.drawContours(mask, img_contours, len(img_contours)-1, 255, -1)
#Subtracción del fondo
new_img =cv2.bitwise_and(img, img, mask=mask)
#Mostrar resultado
cv2.imshow("New image", new_img)
cv2.waitKey(0)

#Genera la segunda segmentación
x, y, w, h = cv2.boundingRect(i)
tercia_h = int(w/3)
tercia_v = int(h/3)

dorsal = new_img[y+tercia_v: y-tercia_v+h, x+tercia_h: x+w-tercia_h]
new_img = new_img[y: y+h, x: x+w]

cv2.imshow("Dorsal", dorsal)
cv2.waitKey(0)

# cv2.Laplacian()
#Guardar imagen
cv2.imwrite("pez70_new.jpg",new_img)
cv2.imwrite("pez70_new_dorsal.jpg",dorsal)