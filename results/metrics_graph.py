import pandas as pd
import matplotlib.pyplot as plt

# TODO - He editat el fitxer de les mètriques manualment (nom del model) per treure la ruta i deixar només el nom
# TODO - S'hauria d'editar la generació del fitxer perquè aquesta informació ja quedi tractada automàticament

# Generar les mètriques del fitxer de log
df = pd.read_csv('../sl_models/training_log.csv', delimiter=',')

# Només per print dataframe
# Mostrar totes les columnes
pd.set_option('display.max_columns', None)
# Mostrar tot el contingut de cada cel·la
pd.set_option('display.max_colwidth', None)
# Ajustar l'ample de la fila
pd.set_option('display.width', None)

# Només es mostra la mètrica final (hi ha guardada la mètrica de cada epoch)
df_final = df[df['epoch'] == 'Final']

# Ordenar per nom del model
df_final = df_final.sort_values(by='nombre_modelo', ascending=False)

# Crear el gràfic
plt.figure(figsize=(18, 8))

# Ample de la barra
bar_width = 0.2

x = range(len(df_final))

# Dibuixar les barres
# plt.barh(x, df_final['loss'], height=bar_width, label='Loss', color='blue')
# plt.barh([p + bar_width for p in x], df_final['accuracy'], height=bar_width, label='Accuracy', color='green')
plt.barh(x, df_final['accuracy'], height=bar_width, label='Accuracy', color='green')
plt.barh([p + bar_width for p in x], df_final['precision'], height=bar_width, label='Precision', color='red')
plt.barh([p + bar_width*2 for p in x], df_final['recall'], height=bar_width, label='Recall', color='orange')

# Afegir etiquetes i títol del gràfic
plt.xlabel('Valors')
plt.ylabel('Models')
plt.title('Mètriques de cada model')
plt.yticks([p + bar_width*1.5 for p in x], df_final['nombre_modelo'])

# Afegir llegenda
plt.legend()

# Ajustar els marges per evitar que els noms del eix Y es tallin
plt.tight_layout()

# Emmagatzemar i mostrar el gràfic
plt.savefig('sl-metriques.png')
plt.show()
