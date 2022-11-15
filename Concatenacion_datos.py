import numpy as np
agregar_mas: bool = 1

datos = []
# datos = np.asanyarray(datos)
while agregar_mas:
    file = input("Indique la ruta del archivo: ")
    try:
        file_data = np.load(file)
        file_data = np.asanyarray(file_data)

        for dato in file_data:
            datos.append(dato)
    except:
        print("No fue posible encontrar la ruta especificada\n\n")
        continue
    
        # datos = datos.concatenate(file_data)
    agregar_mas = bool(int(input("Agregar mas datos (0 - NO, 1 - SI): ")))

datos = np.asanyarray(datos)
nombre = str(input("Nombre de archivo nuevo: "))

np.save("{}{}".format(nombre, ".npy"), datos)

print(datos.shape)