import streamlit as st
import streamlit.components.v1 as stc
from streamlit_modal import Modal
import matplotlib.pyplot as plt

# File Processing Pkgs
import pandas as pd
import docx2txt
from PIL import Image
from streamlit_pdf_viewer import pdf_viewer

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import load

@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img


def main():
    st.title("SKIILS")



    menu = ["Home", "Dataset", "Models", "About", "Instruction"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        link = "https://media.tenor.com/Km11GYbvYY0AAAPo/good-morning.mp4"
        st.video(link, autoplay = True, loop = True)



    elif choice == "Dataset":
        st.subheader("Dataset")
        data_file = st.file_uploader("Upload CSV", type=['csv'])
        if st.button("Process"):
            if data_file is not None:
                file_details = {"Filename": data_file.name, "FileType": data_file.type, "FileSize": data_file.size}
                st.write(file_details)

                startDataFrame = pd.read_csv(data_file)
                startDataFrame = startDataFrame.drop(index=0)
                st.session_state['startDataFrame'] = startDataFrame
                st.dataframe(startDataFrame)
        if st.button("Prepare data"):
            startDataFrame = st.session_state['startDataFrame']
            features = startDataFrame.loc[:, startDataFrame.columns != 'label']
            labels = pd.read_csv("C:/Users/skills/Desktop/3/training_datasets/sample_submission.csv")

            scaler = StandardScaler()
            # Fit only on X_train
            scaler.fit(features)

            # Scale both X_train and X_test
            features = scaler.transform(features)
            st.session_state['features'] = features
            st.session_state['labels'] = labels

            st.header("Данные подготовленны")
            st.video("https://media.tenor.com/1kf4J-xC_68AAAPo/cat.mp4", autoplay = True, loop = True)


    elif choice == "Models":
        st.subheader("Models")
        from sklearn.metrics import accuracy_score
        models_handbook = ["Home", "KNeighborsClassifier", "GaussianNB", "DecisionTreeClassifier"]
        choice = st.sidebar.selectbox("Choose model", models_handbook)
        if choice == "Home":
            st.video("https://media.tenor.com/K8FMp--E6kgAAAPo/cats-funny.mp4", autoplay=True, loop=True)
        if choice == "KNeighborsClassifier":
            st.subheader("KNeighborsClassifier")
            features = st.session_state['features']
            labels = st.session_state['labels']
            knn = load('C:/Users/skills/Desktop/5/knn.joblib')
            pred = knn.predict(features)
            score = accuracy_score(labels, pred)
            st.text("Предсказание выполнено")
            st.text(score)
            plt.figure(figsize=(8, 8))
            #plt.scatter(features, labels, color='blue', label='Actual')
            #plt.scatter(features, pred, color='red', label='Predicted')
            plt.title('KNN')
            plt.xlabel('1')
            plt.ylabel('2')
            st.pyplot(plt.gcf())

        if choice == "GaussianNB":
            st.subheader("GaussianNB")
            from sklearn.naive_bayes import GaussianNB
            features = st.session_state['features']
            labels = st.session_state['labels']
            gaus = load('C:/Users/skills/Desktop/5/gaus.joblib')
            pred = gaus.predict(features)
            score = accuracy_score(labels, pred)
            st.text("Предсказание выполнено")
            st.text(score)
        if choice == "DecisionTreeClassifier":
            st.subheader("DecisionTreeClassifier")
            st.image("https://media.tenor.com/pIzpROAohx0AAAAj/hamstercombat-hamsterkobmat.gif")
            features = st.session_state['features']
            labels = st.session_state['labels']
            tree = load('C:/Users/skills/Desktop/5/tree.joblib')
            pred = tree.predict(features)
            score = accuracy_score(labels, pred)
            st.text("Предсказание выполнено")
            st.text(score)

    elif choice == "About":
        st.subheader("About")
        modal = Modal(
            "Demo Modal",
            key="demo-modal",
            padding=20,  # по умолчанию
            max_width=744  # по умолчанию
        )
        open_modal = st.button("Open")
        if open_modal:
            modal.open()
        if modal.is_open():
            with modal.container():

                st.text("Фото Автора")
                st.video("https://media.tenor.com/KiUfiJ7cvc4AAAPo/cat-sombrero.mp4",autoplay = True, loop = True)
                st.text("Интерфейс создал Мельников Иван")
                st.text("Организация ИГЭУ")
    elif choice == "Instruction":
        st.subheader("Instruction")
        pdf_viewer("C:/Users/skills/Desktop/5/6.pdf")


if __name__ == '__main__':
    main()