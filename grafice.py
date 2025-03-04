import matplotlib.pyplot as plt
from scikitplot.metrics import (plot_confusion_matrix, plot_roc, plot_cumulative_gain,plot_lift_curve)

def plot_matrice_confuzie(y, predictie, nume_model):
    fig=plt.figure(figsize=(7,6))
    ax=fig.add_subplot(1,1,1)
    plot_confusion_matrix(y,predictie,title="Matricea de confuzie - "+nume_model,ax=ax,normalize=True)
    plt.savefig("out/"+nume_model+"_cm.png")

def show():
    plt.show()

def plot_grafice_evaluare(y,proba,nume_model):
    fig = plt.figure(figsize=(12,6))
    ax = fig.subplots(1,3)
    plot_roc(y,proba,title="Plot ROC - "+nume_model,ax=ax[0])
    plot_cumulative_gain(y,proba,title="Plot Gain - "+nume_model,ax=ax[1])
    plot_lift_curve(y,proba,title="Plot Lift - "+nume_model,ax=ax[2])
    plt.savefig("out/"+nume_model+"_RocGainLift.png")


def plot_distributie_discriminanta(model, x_testare, y_testare):
    proba = model.predict_proba(x_testare)
    plt.figure(figsize=(10, 6))
    plt.scatter(proba[:, 0], proba[:, 1], c=y_testare, cmap='viridis', alpha=0.8)
    plt.xlabel('Probabilitate clasa 0')
    plt.ylabel('Probabilitate clasa 1')
    plt.title('Distribuția în axe discriminante pentru modelul Naive Bayes')
    plt.colorbar()
    plt.savefig("out/DistAxeDis_Bayes.png")

