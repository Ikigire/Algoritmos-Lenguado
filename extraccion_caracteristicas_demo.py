import cv2
import easyocr
import numpy as np
from OperacionesIMG import reduccionEscalaGrises

#Leer imagen
image = cv2.imread("pez70.JPG")
# image = cv2.resize(image, (500, 500), interpolation=cv2.INTERSECT_NONE)
#Mostrar imagen
cv2.imshow("Original imagen",image)
cv2.waitKey(0)

#Filas y columnas de la imagen
print('image.shape=',image.shape)

#Recortar una imagen
img = image[0:445,0:615]

#cv2.imshow('Imagen recortada',img)

#Convertir a escala de grises
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Mostrar imagen
cv2.imshow("Gray Scale",gray_img)
cv2.waitKey(0)  
gray_img = cv2.GaussianBlur(gray_img, (3,3), 2)
reduccionEscalaGrises(gray_img)
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

i = img_contours[-1]
# for i in img_contours:
# 	if cv2.contourArea(i) > 400:
# 		break

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

longitud_pez = w
altura_pez = h

dorsal = new_img[y+tercia_v:y+h-tercia_v, x+tercia_h:x+w-tercia_h]

cv2.imshow("Dorsal", dorsal)
cv2.waitKey(0)

size = list(dorsal.shape[:2])
print ("Tamaño = ", size)
size[0] *= 4
size[1] *= 4
longitud_dorsal = size[1]
altitud_dorsal = size[0]
dorsal = cv2.resize(dorsal, tuple(reversed(size)), interpolation= cv2.INTER_AREA)

new_img = new_img[y: y+h, x: x+w]

cv2.imshow("New image", new_img)
cv2.waitKey(0)

cv2.imshow("Dorsal", dorsal)
cv2.imshow("Dorsal 2", dorsal.copy())
cv2.waitKey(0)

#copia del dorsal
# dorsal_filtrado = dorsal.copy()

#reducción de ruido
# dorsal_filtrado = dorsal
dorsal_filtrado = cv2.GaussianBlur(dorsal, (3,3), 2)

cv2.imshow("Dorsal filtrado", dorsal_filtrado)
cv2.imshow("Dorsal filtrado 2", dorsal_filtrado)
cv2.waitKey(0)

dorsal_filtrado = cv2.cvtColor(dorsal_filtrado, cv2.COLOR_BGR2GRAY)
cv2.imshow("Dorsal filtrado", dorsal_filtrado)
cv2.imshow("Dorsal filtrado 2", dorsal_filtrado)
cv2.waitKey(0)

dorsal_filtrado2 = dorsal_filtrado.copy()

# dorsal_filtrado = reduccionEscalaGrises(dorsal_filtrado, 35, 65, 90)
# dorsal_filtrado2 = reduccionEscalaGrises(dorsal_filtrado2, 100,155,215)
# cv2.imshow("Dorsal filtrado 2", dorsal_filtrado2)
# cv2.imshow("Dorsal filtrado", dorsal_filtrado)
# cv2.waitKey(0)

# dorsal_filtrado = cv2.adaptiveThreshold(dorsal_filtrado, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 9)
dorsal_filtrado = cv2.adaptiveThreshold(dorsal_filtrado, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 55, 7)
# dorsal_filtrado2 = cv2.adaptiveThreshold(dorsal_filtrado2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY_INV, 601, 31)
# _, dorsal_filtrado2 = cv2.threshold(dorsal_filtrado2, 20, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_MASK)
# _, dorsal_filtrado2 = cv2.threshold(dorsal_filtrado2, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRUNC)
_, dorsal_filtrado2 = cv2.threshold(dorsal_filtrado2, 105, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRUNC)

#Aplicando filtro Laplaciano
# dst = cv2.Laplacian(dorsal_filtrado, cv2.CV_16S, ksize=3)
# Reconvirtiendo a UInt8
# dorsal_filtrado = cv2.convertScaleAbs(dst)

cv2.imshow("Dorsal filtrado", dorsal_filtrado)
cv2.imshow("Dorsal filtrado 2", dorsal_filtrado2)
cv2.waitKey(0)

# Conteo de pigmentos Negros
pigmentos_negros = cv2.findContours(dorsal_filtrado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
cv2.contourArea
def filter_func(contour):
	if cv2.contourArea(contour) > 45:
		return True
	else:
		return False

pigmentos_negros = list(filter(lambda contour: cv2.contourArea(contour) > 46, pigmentos_negros))

cv2.drawContours(image=dorsal, contours=pigmentos_negros, contourIdx=-1, color=(0,0,0), thickness=1, lineType=cv2.LINE_AA)

# Conteo de pigmentos Blancos
pigmentos_blancos = cv2.findContours(dorsal_filtrado2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

pigmentos_blancos = list(filter(filter_func, pigmentos_blancos))
cv2.drawContours(image=dorsal, contours=pigmentos_blancos, contourIdx=-1, color=(255,255,255), thickness=1, lineType=cv2.LINE_AA)

cv2.imshow("Dorsal", dorsal)
cv2.waitKey(0)

print("\n\nLa cantidad de pigmentos NEGROS encontrados es: {}".format(len(pigmentos_negros)))
print("\n\nLa cantidad de pigmentos BLANCOS encontrados es: {}".format(len(pigmentos_blancos)))

# exit()
# cv2.Laplacian()
#Guardar imagen
cv2.imwrite("pez70_new.jpg",new_img)
cv2.imwrite("pez70_new_dorsal.jpg",dorsal)

###########################################""
#  |                                      	|
#  |                                      	|
#  |  Detección de la escala en pixeles   	|
#  |		Y lectura del texto				|
#  |										|
#  |										|
###########################################""

#Recortar de la imágen para eliminar el pez
img = image[405::,0::]

#cv2.imshow('Imagen recortada',img)

#Convertir a escala de grises
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Mostrar imagen
cv2.imshow("Gray Scale",gray_img)
cv2.waitKey(0)

#Generar imagen binaria
_, thresh = cv2.threshold(gray_img, 253, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#Mostrar Imagen
cv2.imshow("Binary Image",thresh)
cv2.waitKey(0)

#Detección de contornos
#img_contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
img_contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
img_contours = sorted(img_contours, key=cv2.contourArea)

# cv2.drawContours(gray_img, img_contours, -1, 170, 2)
cv2.imshow("Gray Scale",gray_img)
cv2.waitKey(0)

contorno = img_contours[0]
#Volviendo a recortar la imágen para tratar de omitir la región en blanco y el tecto que tiene la escala
print("\nContornos encontrados (linea 157) = ", len(img_contours))
if len(img_contours) > 1:
	length = img.shape[2]
	menor = length
	contorno = img_contours[0]
	for c in img_contours:
		x,y,w,h = cv2.boundingRect(c)
		nuevo_valor = length-x
		if nuevo_valor < menor and w > 80 and h > 10:
			contorno = c
			menor = nuevo_valor
#Con esta linea logras obtener los atributos de la longitud
x,y,w,h = cv2.boundingRect(contorno)
gray_img_text = gray_img[y: y+h, x: x+w]
gray_img = gray_img[y+2:int(y+h/2)-2,x+2:x+w-2]



#Volvemos a repetir el proceso completo pero ahora la finalidad es obtener únicamente la linea que representa la escala
cv2.imshow("Gray Scale for text",gray_img_text)
cv2.waitKey(0)

cv2.imshow("Gray Scale",gray_img)
cv2.waitKey(0)

# Reescalando la imagen para poder realizar la lectura del texto sin mayor complicación
size = list(gray_img_text.shape[:2])
print ("Tamaño = ", size)
size[0] *= 5
size[1] *= 5

gray_img_text = cv2.resize(gray_img_text, tuple(reversed(size)), interpolation= cv2.INTER_AREA)
gray_img_text = reduccionEscalaGrises(gray_img_text, 80, 0,0)
cv2.imshow("Gray Scale for text",gray_img_text)
cv2.waitKey(0)

# Realizando la detección del texto de la escala
reader = easyocr.Reader(['es'], gpu=False)
textos = reader.readtext(gray_img_text)

escala = ""
unidad_medida = ""
texto_escala = textos[0][1]

for char in texto_escala:
	if char in "0123456789":
		escala += char
	else:
		unidad_medida += char
escala = int(escala)
unidad_medida = unidad_medida.strip()
print("Unidad de medida encontrada:",unidad_medida.strip())
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
print("Contornos encontrados (linea 214) = ", len(img_contours))

#Con esta linea logras obtener los atributos de la longitud
#   x=pixel donde empieza en el eje horizontal, 
#   y=pixel donde empieza en el eje vertical,
#   w=ancho en pixeles de la imagen,
#   h=alto en pixeles de la imagen
x,y,w,h = cv2.boundingRect(img_contours[-1])
proporcion = w/escala
print("\n\nLa escala está dada por la proporcion: {} pixeles = {} {}; es decir: {} = {} {}".format(w,escala, unidad_medida, proporcion, 1, unidad_medida))
print("\nLongitud del pez: {} {}\n\nAltura del pez:{} {}\n\nMedida de imágen dorsal: {} x {} {}".format((longitud_pez/proporcion),unidad_medida, (altura_pez/proporcion),unidad_medida, (longitud_dorsal/(proporcion*3)),(altitud_dorsal/(proporcion*3)), unidad_medida))
print("El pez pesa aproximadamente {} g".format((longitud_pez/proporcion) * ((altura_pez/proporcion*2) * (altura_pez/proporcion*2)) / 29))
