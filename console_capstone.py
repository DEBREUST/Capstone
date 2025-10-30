import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

#___________________________________
# Fichier pour stocker les resultats de predictions 

csv_file = "historique_predictions.csv"

#______________________
# Chargement du modèle

try:
    model = joblib.load("xgb_model.pkl")
except Exception as e:
    st.error(f"Impossible de charger le modèle : {e}")
    st.stop()

# ________________________________
# Info importantes pour le modèle

st.title("Prédiction du Risque de Maladie Cardiovasculaire") # Titre
st.header("Veuillez remplir les informations suivantes comme indiqué :")   # En tete

# ___________________________________________________
# Collecte des informations utilisateur et conversion

age = st.number_input("Âge (en années)", 
    min_value=30, 
    max_value=65, 
    help="Veuillez entrer un âge compris entre 30 et 65 ans!")
gender = st.selectbox("Sexe", ["Homme", "Femme"])
height = st.number_input("Taille (cm)", 
    min_value=55,
    max_value=250,
    help="Veuillez entrer une hauteur comprise entre 55 et 250 cm !")
weight = st.number_input("Poids (kg)",
    min_value= 10,
    max_value=200,
    help="Veuillez entrer un poids compris entre 10 et 200 kg !")
smoke = st.selectbox("Fumeur ?", ["Oui", "Non"])
alcool = st.selectbox("Buveur ?", ["Oui", "Non"])
p_systole = st.number_input("Pression Systolique (mmHg)",
    min_value= 60, 
    max_value=240, 
    help="Veuillez entrer une valeur comprise entre 60 et 240 cm !")
p_diastole = st.number_input("Pression Diastolique (mmHg)", 
    min_value=40, 
    max_value=180, 
    help="Veuillez entrer une valeur comprise entre 40 et 180 cm !")
cholesterol = st.selectbox("Cholestérol", ["Normal", "Élevé", "Très élevé"])
glucose = st.selectbox("Glucose", ["Normal", "Élevé", "Très élevé"])
active = st.selectbox("Activité physique ?", ["Oui", "Non"])

#Conversion des données en numériques

gender = 1 if gender == "Femme" else 2
smoke = 1 if smoke == "Oui" else 0
alcool = 1 if alcool == "Oui" else 0
active = 1 if active == "Oui" else 0
cholesterol = {"Normal": 1, "Élevé": 2, "Très élevé": 3}[cholesterol]
glucose = {"Normal": 1, "Élevé": 2, "Très élevé": 3}[glucose]


# Calcul et catégorisation de l'IMC

height_m = height / 100
imc = weight / (height_m ** 2)

if imc < 18.5:
    imc_cat = 0
elif imc <= 24.9:
    imc_cat = 1
elif imc <= 29.9:
    imc_cat = 2
else:
    imc_cat = 3


# ________________________________________________
# Structure des données
# Noms des colonnes dans l'ordre de l'entraînement
feature_names = ['age', 'gender', 'p_systole', 'p_diastole', 'cholesterol',
                 'glucose', 'smoke', 'alcool', 'active', 'imc_cat']

# Vecteur d'entrée
input_values = [age, gender, p_systole, p_diastole, cholesterol, glucose,
                smoke, alcool, active, imc_cat]

features = pd.DataFrame([input_values], columns=feature_names)

# _______________________
# Lecture de l'historique

if os.path.exists(csv_file):
    df_existing = pd.read_csv(csv_file)
else:
    df_existing = pd.DataFrame()


# Bouton de prédiction
# ______________________

if st.button("Prédire le risque"):

    # Prédiction
    prediction = model.predict(features)[0]

    try:
        proba = model.predict_proba(features)[0,1] * 100
    except AttributeError:
        proba = 0000.0
        st.warning("Désolé ce modèle n'arrive pas a fournir la probabilité pour vos données.")

    # Affichage du résultat
    if prediction == 1:
        st.error(f"Risque détecté ! Probabilité estimée : {proba:.2f}%")
        st.info("Cette prédiction n'est pas fiable à 100 % , mais dépasse bien les 70%.  \n Veuillez consulter un médecin pour plus de détails cliniques!  \n Bon courage et prenez soin de vous! ")
    else:
        st.success(f"Aucun risque détecté. Probabilité estimée : {proba:.2f}%")

    # Enregistrement dans le CSV
    data = {
        "age": age,
        "gender": gender,
        "p_systole": p_systole,
        "p_diastole": p_diastole,
        "cholesterol": cholesterol,
        "glucose": glucose,
        "smoke": smoke,
        "alcool": alcool,
        "active": active,
        "imc_cat": imc_cat,
        "prediction": prediction,
        "proba_risque_%": round(proba, 2)
    }

    df_new = pd.DataFrame([data])
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv(csv_file, index=False)
    st.info(f" Vos données ont été enregistrées !\n Vous pouvez les consulter dans le fichier ci-dessous")


    # Affichage de l'historique des dernières prédictions
    with st.expander("Historique des prédictions récentes"):
        st.dataframe(df_combined.tail(5))
