from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

def plot_conf_matrix(y_true, y_pred, class_names=['Speech', 'Music']):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True label',
           xlabel='Predicted label')
    plt.title("Confusion Matrix")
    plt.show()

def print_class_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, target_names=['Speech', 'Music'])
    print(report)

