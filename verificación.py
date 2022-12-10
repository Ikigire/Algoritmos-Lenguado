import numpy as np

x_d = np.load("X_data.npy")
y_d = np.load("Y_data.npy")

print("Datos de prueba :\n\t 0 :{} - {}\n\t 1219 :{} - {}\n\t 1220 :{} - {}\n\t {} :{} - {}".format(x_d[0].shape, y_d[0],x_d[1219].shape, y_d[1219],x_d[1220].shape, y_d[1220],len(x_d)-1, x_d[len(x_d)-1].shape, y_d[len(x_d)-1],))
print(x_d.shape, y_d.shape)

# x_d = np.load("X_data_bg.npy")
# y_d = np.load("Y_data_bg.npy")

# print("Datos de prueba :\n{} - {}\n{} - {}\n{} - {}\n{} - {}".format(x_d[0].shape, y_d[0],x_d[609].shape, y_d[609],x_d[610].shape, y_d[610],x_d[1038].shape, y_d[1038],))