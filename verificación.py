import numpy as np

x_d = np.load("X_data_no_bg.npy")
y_d = np.load("Y_data_no_bg.npy")

print("Datos de prueba :\n{} - {}\n{} - {}\n{} - {}\n{} - {}".format(x_d[0].shape, y_d[0],x_d[609].shape, y_d[609],x_d[610].shape, y_d[610],x_d[1038].shape, y_d[1038],))

x_d = np.load("X_data_bg.npy")
y_d = np.load("Y_data_bg.npy")

print("Datos de prueba :\n{} - {}\n{} - {}\n{} - {}\n{} - {}".format(x_d[0].shape, y_d[0],x_d[609].shape, y_d[609],x_d[610].shape, y_d[610],x_d[1038].shape, y_d[1038],))