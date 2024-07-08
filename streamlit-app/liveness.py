import streamlit as st
import numpy as np
import cv2
import io
import time
import tensorflow as tf
from tensorflow.keras.models import load_model

import os

# Caminho absoluto para o diretório onde o script está localizado
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, '..', 'modelos', 'modelX.keras')

print("script_dir", script_dir)
print("model_path", model_path)

# Listar diretórios e arquivos para depuração
st.write("Conteúdo do diretório atual:")
st.write(os.listdir(script_dir))
st.write("Conteúdo do diretório modelos:")
st.write(os.listdir(os.path.join(script_dir, '..', 'modelos')))

print("Keras version:", tf.keras.__version__)

"""
# Computer Vision
## Liveness Detection

O detector de Liveness (Vivacidade) tem por objetivo estabelecer um índice que atesta o quão 
confiável é a imagem obtida pela câmera.
Imagens estáticas, provindas de fotos manipuladas, são os principais focos de fraude neste tipo de validação.
Um modelo de classificação deve ser capaz de ler uma imagem da webcam, classificá-la como (live ou não) e 
exibir sua probabilidade da classe de predição.

"""

# Carregar o modelo salvo
# Exibe uma animação de carregamento enquanto o modelo está sendo carregado

print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

with st.spinner('Carregando o modelo...'):
    model = load_model(model_path)
    # Simula um tempo de carregamento
    time.sleep(2)

# Informa que o modelo foi carregado com sucesso
st.success('Modelo carregado com sucesso!')

# Função para pré-processar a imagem
def preprocess_image(image):
    try:
        print("Pré-processando a imagem...")
        image = cv2.resize(image, (224, 224))  # Redimensionar para o tamanho esperado pelo modelo
        print(f"Imagem redimensionada para: {image.shape}")
        image = image.astype('float32') / 255.0  # Normalizar os pixels para o intervalo [0, 1]
        print(f"Imagem normalizada. Valores mínimo e máximo: {image.min()}, {image.max()}")
        image = np.expand_dims(image, axis=0)  # Adicionar uma dimensão extra para representar o lote
        print(f"Dimensão extra adicionada. Forma final da imagem: {image.shape}")
        return image
    except Exception as e:
        print(f"Erro ao pré-processar a imagem: {e}")

uploaded_file = st.file_uploader('Tente uma outra imagem', type=["png", "jpg"])
if uploaded_file is not None:
    img_stream = io.BytesIO(uploaded_file.getvalue())
    imagem = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), 1)
    st.image(imagem, channels="BGR")


camera = st.camera_input("Tire sua foto", help="Lembre-se de permitir ao seu navegador o acesso a sua câmera.")

if camera is not None:
    bytes_data = camera.getvalue()
    imagem = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)


if camera or uploaded_file:

    with st.spinner('Classificando imagem...'):
        # Pré-processar a imagem
        image_processed = preprocess_image(imagem)
        
    
        # Fazer a previsão
        try:
            prediction = model.predict(image_processed)
            st.success("Previsão feita com sucesso.")
            
            # Considerando que a saída é uma probabilidade, podemos interpretar a previsão
            vivacidade_percentual = prediction[0][0] * 100  # Converter para percentual         
        
            if vivacidade_percentual > 50:
                st.success(f"Tipo da foto: Real")
                st.success(f"Percentual de vivacidade: {vivacidade_percentual:.6f}%")                
            else:
                st.error(f"Tipo da foto: Fraude")
                st.error(f"Percentual de vivacidade: {vivacidade_percentual:.6f}%")
                st.error("""
                        Esta foto pode ser uma tentativa de fraude.
                        Se você não concorda com esta informação, use um lugar mais iluminado para sua foto,
                        ou use uma câmera com menos ruído.
                        Se for upload de foto, tenha certeza que ela possui boa nitidez com foco na face e 
                        fundo limpo, de preferência monocromático.
                        """)                    

       
                
                
        except Exception as e:
            st.error(f"Erro ao fazer a previsão: {e}")
    
