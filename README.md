# README - Proiect - Predictia Diabetului

## Despre Dataset
Acest set de date provine de la Institutul Național de Diabet, Digestiv și Boli Renale. Scopul acestui proiect este de a prezice diagnosticul de diabet zaharat pe baza unor măsurători specifice incluse în dataset. Datele conțin informații despre femei cu vârsta de cel puțin 21 de ani și de moștenire indiană.

- **Tipul studiului:** Clasificare
- **Sursa datelor:** [Kaggle - Diabetes Dataset](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset/data)
- **Fișiere utilizate:**
  - `diabetes.csv` - setul de antrenare și testare (538 observații)
  - `diabetes_apply.csv` - setul de aplicare (230 observații)

**Variabilă țintă:** `Outcome` - indică prezența diabetului.

## Analiza indicatorilor WOE și IV

Indicatorii Weight of Evidence (WOE) și Information Value (IV) sunt folosiți pentru a determina influența fiecărei variabile asupra predicției diabetului. Principalele concluzii:
- Cea mai mare influență o are **Glucose** (IV = 1.35), urmat de **BMI** (IV = 0.63).
- Variabilele cu IV peste 0.02 au fost selectate pentru modelele de machine learning.

## Modelele utilizate și performanța acestora

### 1. **Naive Bayes**
### 2. **Arbori de Decizie**
### 3. **Random Forest**
### 4. **Regresia Logistică** *(Modelul optim)*
### 5. **k-Nearest Neighbors (kNN)**
### 6. **Support Vector Machine (SVM)**

## Rezultate și fișiere generate
- **Predicții pentru setul de testare:** `Predictii_test.csv`
- **Predicții pentru setul de aplicare:** `Predictii.csv`
- **Clasificări eronate pentru modelul optim:** `Clasificari_eronate.csv`

## Concluzii
- Regresia Logistică oferă cele mai bune rezultate în predicția diabetului.
- Modelele Random Forest și SVM sunt alternative solide cu performanțe similare.
- Modelul Arbori de Decizie are cea mai slabă acuratețe, fiind mai puțin potrivit pentru acest set de date.

