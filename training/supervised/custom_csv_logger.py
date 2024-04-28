import csv

from tensorflow.keras.callbacks import CSVLogger


class CustomCSVLogger(CSVLogger):
    def __init__(self, model_name: str, final_model_name: str, filename: str, separator: str = ',', append: bool = False):
        super().__init__(filename, separator=separator, append=append)
        self.model_name = model_name
        self.final_model_name = final_model_name

    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get('accuracy')
        loss = logs.get('loss')
        precision = logs.get('precision')
        recall = logs.get('recall')
        f1_score = logs.get('f1_score')

        # Abrir el archivo CSV en modo de añadir ('a')
        with open(self.filename, 'a', newline='') as file:
            writer = csv.writer(file)

            # Escribir las métricas en una nueva fila
            writer.writerow([self.model_name, self.final_model_name, epoch, accuracy, loss, precision, recall, f1_score])