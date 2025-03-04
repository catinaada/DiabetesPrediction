from functii import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


set_antrenare_testare = pd.read_csv("in/diabetes.csv",index_col=0)
set_aplicare = pd.read_csv("in/diabetes_apply.csv",index_col=0)

nan_replace(set_antrenare_testare)
nan_replace(set_aplicare)

# print(set_antrenare_testare)
# print(set_aplicare)

#Impartim setul de testare in doua
variabile_obs = list(set_antrenare_testare)
predictori = variabile_obs[:-1]
tinta = variabile_obs[-1]
# print("Predictorii sunt: ")
# print(predictori)
# print("Variabila tinta este: ")
# print(tinta)

t_woe, t_iv = woe_iv(set_antrenare_testare, predictori, tinta)
predictori_ = list(t_iv.loc[t_iv['IV'] > 0.02, 'Variabila'])
print("Predictorii sunt: ")
print(predictori_)


x_antrenare, x_testare, y_antrenare, y_testare=(
    train_test_split(set_antrenare_testare[predictori_],set_antrenare_testare[tinta],test_size=0.3))
print(x_antrenare)
print(x_testare)
print(y_antrenare)
print(y_testare)

#Tabel predictii:
tabel_predictii_test = pd.DataFrame({
    tinta: y_testare}, index=x_testare.index)

model_optim = None
index_kappa = 0

#Creare si analizare model bayessian
model_b = GaussianNB()
kappa = analiza_model(model_b,x_antrenare,y_antrenare,x_testare,y_testare,tabel_predictii_test,nume_model="Bayes")

if kappa > index_kappa:
    index_kappa = kappa
    model_optim = model_b

plot_distributie_discriminanta(model_b, x_testare, y_testare)


# Creare si analizare - Arbori de decizie
model_dt = DecisionTreeClassifier()
kappa = analiza_model(model_dt, x_antrenare, y_antrenare, x_testare, y_testare, tabel_predictii_test, nume_model="DT")

if kappa > index_kappa:
    index_kappa = kappa
    model_optim = model_dt

# Creare si analizare - RF
model_rf = RandomForestClassifier()
kappa = analiza_model(model_rf, x_antrenare, y_antrenare, x_testare, y_testare, tabel_predictii_test, nume_model="RF")

if kappa > index_kappa:
    index_kappa = kappa
    model_optim = model_rf

# Creare si analizare - RL SAU RIDGE
if len(model_b.classes_) == 2:
    model_regresie_logistica = LogisticRegression()
    kappa = analiza_model(model_regresie_logistica, x_antrenare, y_antrenare, x_testare, y_testare, tabel_predictii_test, nume_model="RL")
    if kappa > index_kappa:
        index_kappa = kappa
        model_optim = model_regresie_logistica
else:
    model_ridge = RidgeClassifier(max_iter=2000)
    kappa = analiza_model(model_ridge, x_antrenare, y_antrenare, x_testare, y_testare, tabel_predictii_test, nume_model="RIDGE")
    if kappa > index_kappa:
        index_kappa = kappa
        model_optim = model_ridge

# Creare si analizare - KNN
model_knn = KNeighborsClassifier()
kappa = analiza_model(model_knn, x_antrenare, y_antrenare, x_testare, y_testare, tabel_predictii_test, nume_model="knn")
if kappa > index_kappa:
    index_kappa = kappa
    model_optim = model_knn

# Creare si analizare - svc

model_svm = SVC(probability=True)
kappa = analiza_model(model_svm, x_antrenare, y_antrenare, x_testare, y_testare, tabel_predictii_test, nume_model="SVM")
if kappa > index_kappa:
    index_kappa = kappa
    model_optim = model_svm

tabel_predictii_test.to_csv("out/Predictii_test.csv")

print("Modelul optim este: ")
print(model_optim)

#predictii set aplicare
set_aplicare["Predictie"] = model_optim.predict(set_aplicare[predictori_])
set_aplicare.to_csv("out/Predictii.csv")

# Calculare si afișare tabel clasificari eronate pentru modelul optim
if 'GaussianNB' in str(model_optim):
    tabel_clasificari_eronate(y_testare, tabel_predictii_test["Bayes_Predictie"], model_b.classes_)
elif "DecisionTreeClassifier" in str(model_optim):
    tabel_clasificari_eronate(y_testare, tabel_predictii_test["DT_Predictie"], model_dt.classes_)
elif 'RandomForestClassifier' in str(model_optim):
    tabel_clasificari_eronate(y_testare, tabel_predictii_test["RF_Predictie"], model_rf.classes_)
elif "LogisticRegression" in str(model_optim):
    tabel_clasificari_eronate(y_testare, tabel_predictii_test["RL_Predictie"], model_regresie_logistica.classes_)
elif "RidgeClassifier" in str(model_optim):
    tabel_clasificari_eronate(y_testare, tabel_predictii_test["RIDGE_Predictie"], model_ridge.classes_)
elif "KNeighborsClassifier" in str(model_optim):
    tabel_clasificari_eronate(y_testare, tabel_predictii_test["knn_Predictie"], model_knn.classes_)
elif "SVC" in str(model_optim):
    tabel_clasificari_eronate(y_testare, tabel_predictii_test["SVM_Predictie"], model_svm.classes_)
else:
    print("Modelul optim nu este recunoscut.")




