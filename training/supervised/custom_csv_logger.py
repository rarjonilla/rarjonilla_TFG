import csv

from tensorflow.keras.callbacks import CSVLogger


class CustomCSVLogger(CSVLogger):
    """Classe que omple el CSV amb el resultat de les mètriques de l'entrenament"""
    def __init__(self, model_name: str, final_model_name: str, filename: str, separator: str = ',', append: bool = False):
        super().__init__(filename, separator=separator, append=append)
        # Dataset utilitzat
        self.model_name = model_name
        # Nom del model
        self.final_model_name = final_model_name

    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get('accuracy')
        loss = logs.get('loss')
        precision = logs.get('precision')
        recall = logs.get('recall')
        # f1_score = logs.get('f1_score')

        # Abrir el archivo CSV en modo de añadir ('a')
        with open(self.filename, 'a', newline='') as file:
            writer = csv.writer(file)

            # Escriure les mètriques en una nova fila
            # writer.writerow([self.model_name, self.final_model_name, epoch, accuracy, loss, precision, recall, f1_score])
            writer.writerow([self.model_name, self.final_model_name, epoch, accuracy, loss, precision, recall])