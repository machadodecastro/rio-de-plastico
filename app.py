import os
import shutil
import time
from collections import Counter
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(layout="wide", page_title="Rio de Plástico")
st.write("# Detectar se há plástico no rio ou não")

@st.cache_data
def load_model():
    '''
    Method used to Load the YOLO model
    '''
    # Loading the best model
    model_path = "runs/detect/train/weights/best.pt"
    return YOLO(model_path)


def get_predictions(model, image) -> Image:
    '''
    Method used to get predictions from the model for the image passed
    '''
    res = model.predict(image, save_txt=True)
    # Plotting the bboxes on the image
    res = res[0].plot(line_width=1)
    # Converting from BGR to RGB
    res = res[:, :, ::-1]

    # Converting the image in png
    res = Image.fromarray(res)
    return res

def get_pred_labels() -> dict:
    '''
    Method to get the predicted labels from text file
    '''
    LABELS = {
        0: 'SACO_PLASTICO', 
        1: 'GARRAFA_PLASTICO', 
        2: 'OUTRO_RESIDUO_PLASTICO', 
        3: 'NAO_RESIDUO'
    }
    results = []
    # Reading the predicted labels.txt file
    results_file_path = "runs/detect/predict/labels.txt"
    with open(results_file_path, 'r') as f:
        lines = f.readlines()

    results.extend(LABELS[int(line[0])] for line in lines)
    count_labels = dict(Counter(results))
    # Removing the labels.txt file
    os.remove(results_file_path)

    return count_labels

with st.sidebar:
    st.title("Rio de Plástico")
    st.sidebar.write("Tente enviar uma imagem para prever se há algum plástico \
 (sacos/garrafas/outros itens de plástico) na imagem ou não.")
    with st.form("my_form"):
        if 'model' not in st.session_state:
            with st.spinner('Carregando o modelo ...'):
                # Loading and saving the model to session state
                model = load_model()
                st.session_state['model'] = model
                success_msg = st.success('Modelo carregado com sucesso!')

            time.sleep(2)
            success_msg.empty()

        uploaded_image = st.file_uploader("Carregue uma Imagem", type=["png", "jpg", "jpeg"])
        submitted = st.form_submit_button("Iniciar")
        st.text("")
        st.image('https://www.ictb.fiocruz.br/sites/default/files/logo.png')

if not submitted or not uploaded_image:
    # Stopping the execution if no image is uploaded
    st.stop()
else:
    try:
        # Converting the streamlit image to PIL Image format
        image = Image.open(uploaded_image)
        predicted_image = get_predictions(st.session_state['model'], image)
        predicted_labels = get_pred_labels()
        tab = '&ensp;'
        predictions = f',{tab}'.join(f'{key} : {str(value)}' for key,value in predicted_labels.items())
        st.info(f"PREDICTIONS{tab} → {tab} {predictions}")
        st.image(predicted_image)
    except FileNotFoundError as e:
        del_dir = 'runs/detect/'
        for fname in os.listdir(del_dir):
            if fname.startswith("predict"):
                shutil.rmtree(os.path.join(del_dir, fname))
        st.warning("Algo deu errado. Por favor, recarregue o aplicativo.")
