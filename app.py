import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Charger le modèle et la liste des features utilisées pendant l'entraînement
model = joblib.load("house_price_model.pkl")
features_used = joblib.load("features_list.pkl")  # Charger les 244 colonnes utilisées à l'entraînement

st.title("Prédiction du Prix des Maisons")

# Création des champs d'entrée pour les principales caractéristiques
surface = st.number_input("Aire", min_value=500, step=50)

chambres = st.number_input("Nombre de chambres", min_value=1, step=1)
sdb = st.number_input("Nombre de salles de bain", min_value=1,  step=1)
garage = st.number_input("Nombre de garages", min_value=0, step=1)
annee_construction = st.number_input("Année de construction", min_value=1900, max_value=2025, step=1)

# Créer un DataFrame avec toutes les colonnes attendues par le modèle
data = pd.DataFrame(columns=features_used)
data.loc[0] = 0  # Initialiser toutes les colonnes à zéro

# Remplir avec les valeurs de l'utilisateur (uniquement si la colonne existe)
if "GrLivArea" in data.columns:
    data["GrLivArea"] = surface
if "BedroomAbvGr" in data.columns:
    data["BedroomAbvGr"] = chambres
if "FullBath" in data.columns:
    data["FullBath"] = sdb
if "GarageCars" in data.columns:
    data["GarageCars"] = garage
if "YearBuilt" in data.columns:
    data["YearBuilt"] = annee_construction

# Bouton pour prédire
if st.button("Prédire le Prix "):
    prediction = model.predict(data)
    st.success(f"Le prix estimé de la maison est : {prediction[0]:,.2f} $")
