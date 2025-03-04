import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from grafice import *


def nan_replace(t):
    assert isinstance(t, pd.DataFrame)
    for v in t.columns:
        if t[v].isna().any():
            if is_numeric_dtype(t[v]):
                t[v].fillna(t[v].mean(), inplace=True)
            else:
                t[v].fillna(t[v].mode()[0], inplace=True)

# Analizam modelele, ceva generic
def analiza_model(model,x,y,x_,y_,predictii_test,nume_model):
    model.fit(x,y)
    predictie = model.predict(x_) #predictia pe setul de testare
    predictii_test[nume_model+"_Predictie"] = predictie
    proba = model.predict_proba(x_)
    # print(predictie)
    # print(proba)
    cm = confusion_matrix(y_,predictie)
    # print(cm)
    clase = model.classes_
    t_cm = pd.DataFrame(cm,clase,clase) #tabelul matricei de confuxie
    t_cm["Acuratete"] = np.diag(cm)*100/np.sum(cm,axis=1) #adaug la matricea de confuzie o coloana, acuratetea
    t_cm.to_csv("out/"+nume_model+"_cm.csv")
    plot_matrice_confuzie(y_,predictie,nume_model)
    plot_grafice_evaluare(y_,proba,nume_model)
    acuratete_globala=np.sum(np.diag(cm))*100/len(y_)
    acuratete_medie=t_cm["Acuratete"].mean()
    kappa = cohen_kappa_score(y_,predictie)
    s_acuratete = pd.Series([acuratete_globala,acuratete_medie,kappa],
                          ["Acuratete Globala","Acuratete Medie","Index Kappa"],name="Acuratete")
    s_acuratete.to_csv("out/"+nume_model+"_acuratete.csv")
    # show()
    return kappa

def woe_iv(t, predictori, tinta, bins=10):
    t_woe = pd.DataFrame()
    t_iv = pd.DataFrame()
    for v in predictori:
        if is_numeric_dtype(t[v]) and len(t[v]) > bins:
            categorii = pd.qcut(t[v], bins, duplicates='drop')
            t_ = pd.DataFrame({"x": categorii, "y": t[tinta]})
        else:
            t_ = pd.DataFrame({"x": t[v], "y": t[tinta]})
        # print("Variabila:",v)
        # print(t_)
        d = t_.groupby(by="x", as_index=False, observed=False).agg(['count', 'sum'])
        d.columns = ['Interval', 'Frecventa', 'Clasa_1']
        d.loc[d['Clasa_1'] == 0, 'Clasa_1'] = 1
        d['Clasa_0'] = d['Frecventa'] - d['Clasa_1']
        d.loc[d['Clasa_0'] == 0, 'Clasa_0'] = 1
        d['P_Clasa_1'] = d['Clasa_1'] / d['Clasa_1'].sum()
        d['P_Clasa_0'] = d['Clasa_0'] / d['Clasa_0'].sum()
        d['WOE'] = np.log(d['P_Clasa_1']) - np.log(d['P_Clasa_0'])
        d['IV'] = (d['P_Clasa_1'] - d['P_Clasa_0']) * d['WOE']
        d.insert(0, "Variabila", v)
        t_iv = pd.concat([t_iv, pd.DataFrame({"Variabila": [v], "IV": [d['IV'].sum()]})])
        # print(d)
        t_woe = pd.concat([t_woe, d])
    # print(t_woe)
    # print(t_iv)
    t_woe.to_csv("out/woe.csv")
    t_iv.to_csv("out/iv.csv")
    return t_woe, t_iv

def tabel_clasificari_eronate(y_true, y_pred, clase):
    cm = confusion_matrix(y_true, y_pred)
    corecte = cm.diagonal()
    eronate = cm.sum(axis=1) - corecte
    tabel_erori = pd.DataFrame({'Numar clasificari corecte': corecte, 'Numar clasificari eronate': eronate},
                               index=clase)
    tabel_erori.to_csv("out/Tabel_Clasificari_Eronate.csv")


