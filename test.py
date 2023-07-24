import h5py
import matplotlib.pyplot as plt

# Charger le fichier h5
f = h5py.File("Result/Transformer_Results/Mapillary/Comparison_200E_dataaug_contrast/transformer.h5", "r")

# Accéder aux métriques d'entraînement
training_metrics = f["training"]

# Récupérer les valeurs de la précision et de la validation
accuracy = training_metrics["accuracy"][:]
validation_accuracy = training_metrics["val_accuracy"][:]

# Fermer le fichier h5
f.close()

# Tracer les courbes de précision et de validation
plt.plot(accuracy, label='Précision')
plt.plot(validation_accuracy, label='Validation')
plt.xlabel('Époque')
plt.ylabel('Précision')
plt.legend()
plt.show()