import numpy as np
import pickle
import pandas as pd
import streamlit as st
import joblib
import time

st.image("https://media.istockphoto.com/photos/rear-view-of-baseball-batter-and-catcher-watching-the-pitch-picture-id1174867119?b=1&k=20&m=1174867119&s=170667a&w=0&h=Lpk2muXoNKWB8dTpak55rqwM1ffEddzgSZsmJeZKEvg=", use_column_width= 'always')

# Title
st.title("Prediksi Gaji Pemain MLB")

# Subtitle
st.markdown("Masukkan data statistik untuk mendapatkan hasil prediksi gaji pemain")

# Add sidebar
st.sidebar.markdown("## Prediksi Gaji Pemain MLB!")
st.sidebar.image("http://cdn.shopify.com/s/files/1/0480/9470/7866/collections/ef26964ae31041325cd9672682c01534.jpg?v=1646869133", width = 200)
st.sidebar.markdown("Bagaimana tim mengevaluasi nilai kontrak pemain?")

st.sidebar.markdown("#### Dibuat Oleh Muhammad Rizky Azzakky")

# input bar 1
difference = st.number_input("Average Salary Difference (in $)")

# input bar 2
age = st.slider('Age', 18, 45, 25)

# input bar 3
hits = st.slider('Hits', 0, 250, 100)

# input bar 4
runs= st.slider('Runs', 0, 200, 50)

# input bar 5
rbi = st.slider('RBIs', 0, 200, 75)

# input bar 6
walks = st.slider('Walks', 0, 250, 50)

# input bar 7
so = st.slider('Strikeouts', 0, 250, 50)

# input bar 8
sb = st.slider('Stolen Bases', 0, 100, 10)

# input bar 9
ops = st.number_input("Enter OPS")

# if button is pressed
if st.button("Submit"):

    # unpickle the batting model
    bb_model = joblib.load("pkl/bb_model.pkl")

    # store inputs into df

    column_names = ['Salary Difference', 'Age', 'H', 'R', 'RBI', 'BB', 'SO', 'SB', 'OPS']
    df = pd.DataFrame([[difference, age, hits, runs, rbi, walks, so, sb, ops]], 
                     columns = column_names)

    # get prediction
    prediction = bb_model.predict(df)

    # convert prediction
    converted = round(np.exp(prediction)[0],0)

    with st.spinner('Calculating...'):
        time.sleep(1)
    st.success('Done!')

    st.dataframe(df)

    # output prediction
    st.header(f"Predicted Player Salary: ${converted:,}")
