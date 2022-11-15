from asyncore import read
import cv2
import easyocr

reader = easyocr.Reader(['es'], gpu=False)

image = cv2.imread('pez70.jpg')

#Recortar de la imágen para eliminar el pez
img = image[405:460, 0:615]

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
    gray_img = gray_img[y+2:int(y+h),x:x+w]

cv2.imshow("Gray Scale",gray_img)
cv2.waitKey(0)

size = list(gray_img.shape[:2])
print ("Tamaño = ", size)
size[0] *= 3
size[1] *= 3
print ("Nuevo Tamaño = ", size)
gray_img = cv2.resize(gray_img, tuple(reversed(size)), interpolation= cv2.INTER_AREA)

print ("Tamaño Final = ", gray_img.shape[:2])
cv2.imshow("Gray Scale",gray_img)

textos = reader.readtext(gray_img)

escala = ""
unidad_medida = ""
texto_escala = textos[0][1]

for char in texto_escala:
	if char in "0123456789":
		escala += char
	else:
		unidad_medida += char
escala = int(escala)

print("Valor escala: ", escala)
cv2.waitKey(0)
cv2.destroyAllWindows()