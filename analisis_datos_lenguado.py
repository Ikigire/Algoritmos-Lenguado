from pickle import NONE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
from extraccion_caracteristicas import extraer_caracteristicas_carpeta


# Lectura del archivo CSV
df = pd.read_csv('datos.csv') # Con los datos de lso pigmentos
# df = pd.read_csv('Datos_lenguados_juveniles.csv') # Sin datos de conteo de pigmentos

# Con los datos de conteo de pigmentos
colum_names = ["NO_pez", "ID_pez", "peso", "Long", "ASDA", "IEDA", "ASFA", "IEFA", "FAC", "TAC", "FANE", "TANE", "glucosa_in", "lactosa_in", "cortisol_in", "glucosa_est", "lactosa_est", "cortisol_est", "opercula", "ojo_ojo", "ojo", "boca", "agalla", "aleta", "nariz", "ojo_cachete", "poro_aleta", "ancho", "ojo_agalla", "agalla_nariz", "agalla_boca", "c_style", "p_negros", "p_blancos"]

# Sin los datos de conteo de pigmentos
# colum_names = ["NO_pez", "ID_pez", "peso", "Long", "ASDA", "IEDA", "ASFA", "IEFA", "FAC", "TAC", "FANE", "TANE", "glucosa_in", "lactosa_in", "cortisol_in", "glucosa_est", "lactosa_est", "cortisol_est", "opercula", "ojo_ojo", "ojo", "boca", "agalla", "aleta", "nariz", "ojo_cachete", "poro_aleta", "ancho", "ojo_agalla", "agalla_nariz", "agalla_boca", "c_style"]

df.columns = colum_names

if("p_blancos" not in colum_names):
    folder = "D:/asm_1/Documents/fotos_coloracion_Yael"
    results = extraer_caracteristicas_carpeta(input_folder=folder, method="sc", save_ouput=False)

    pigmentos_blancos=[]
    pigmentos_negros =[]

    for result in results:
        _, datos_dorsal = result
        if not datos_dorsal is None:
            pigmentos_negros.append(int(datos_dorsal[5]))
            pigmentos_blancos.append(int(datos_dorsal[7]))
        else:
            pigmentos_negros.append(0)
            pigmentos_blancos.append(0)

    # print("\n\n Pigmentos Negros:", pigmentos_negros)
    # print("\n\n Pigmentos blancos:", pigmentos_blancos)

    df["p_negros"] = pigmentos_negros
    df["p_blancos"] = pigmentos_blancos

df2 = df
df2 = df[(df["p_negros"] > 10)]
# df2 = df[(df["p_blancos"] > 0)]
print(df2)


# cols = ["NO_pez", "ID_pez", "peso", "Long", "FAC", "TAC", "FANE", "TANE", "glucosa_in", "lactosa_in", "cortisol_in", "glucosa_est", "lactosa_est", "cortisol_est"]
cols = ["peso", "Long", "FANE", "TANE", "p_negros", "p_blancos", "c_style"]
sns.pairplot(df2[cols], height=1.5)
plt.tight_layout()
plt.show()
cv2.waitKey(0)

cm = np.corrcoef(df2[cols].values.T)
sns.set(font_scale=1)
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 15},yticklabels=cols,xticklabels=cols)
plt.show()


cols = ["peso", "Long", "glucosa_in", "lactosa_in", "cortisol_in", "p_negros", "p_blancos", "c_style"]
cm = np.corrcoef(df2[cols].values.T)
sns.set(font_scale=1)
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 15},yticklabels=cols,xticklabels=cols)
plt.show()

cols = ["peso", "Long", "glucosa_est", "lactosa_est", "cortisol_est", "p_negros", "p_blancos", "c_style"]
cm = np.corrcoef(df2[cols].values.T)
sns.set(font_scale=1)
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 15},yticklabels=cols,xticklabels=cols)
plt.show()
cv2.waitKey(0)

cols = ["opercula", "ojo_ojo", "ojo", "boca", "agalla", "aleta", "nariz", "c_style"]
df.to_csv("datos.csv", index=False)
cm = np.corrcoef(df2[cols].values.T)
sns.set(font_scale=1)
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 15},yticklabels=cols,xticklabels=cols)
plt.show()
cv2.waitKey(0)

cols = ["ojo_cachete", "poro_aleta", "ancho", "ojo_agalla", "agalla_nariz", "agalla_boca", "c_style"]
df.to_csv("datos.csv", index=False)
cm = np.corrcoef(df2[cols].values.T)
sns.set(font_scale=1)
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 15},yticklabels=cols,xticklabels=cols)
plt.show()
cv2.waitKey(0)