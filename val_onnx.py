import cv2
import numpy as np
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import os

from opencv_onnx import OpencvOnnx
from onnx_yolo import OnnxYolo

def get_max_confidence(detections: list):
    max_confidence = 0
    great_detection = None
    for detection in detections:
        if detection['confidence'] > max_confidence:
            great_detection = detection
    return great_detection

def get_images_path(dict_yaml):
    images_path = []
    path = dict_yaml["path"]
    
    with open(f'{path}{dict_yaml["test"]}', 'r') as file:
        images = file.readlines()

    for image in images:
        image_start = image.split('.')[1]
        image_end = image.split('.')[-1]
        image_path = f'{path}{image_start}.{image_end}'
        image_path = image_path.replace('//', '/')
        images_path.append(image_path.strip())

    return images_path

def validation(model: OpencvOnnx, dict_yaml, output_path = "./runs/val/onnx"):
    classes = dict_yaml["names"]
    confusion_matrix = np.zeros((len(classes)+1,len(classes)+1))
    images = get_images_path(dict_yaml)
    
    for image_path in tqdm(images):
        image = cv2.imread( image_path)
        detections = model.detect(image)
        detection = get_max_confidence(detections=detections)
        try:
            predict = int(detection['class_id'])
        except:
            predict = len(classes)
        
        label_path = image_path.replace('.jpg', '.txt')
        label_path = label_path.replace('images', 'labels')

        with open(label_path, 'r') as file:
            current = int(file.readline().split(' ')[0])
        
        confusion_matrix[predict, current] += 1

    # Plotando a matriz de confusão
    labels = [classes[name] for name in classes]  # Labels para as classes
    labels.append('background')

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap='Blues', ax=ax, colorbar=True, values_format='.0f')
    ax.grid(False)  # Removendo os traços de grade
    plt.xticks(rotation=45, ha="right", fontsize=10)  # Rotação e alinhamento
    plt.yticks(fontsize=10)
    plt.xlabel("True label")  # Trocando os rótulos
    plt.ylabel("Predicted label")
    plt.title("Confusion Matrix")

    # Salvando a imagem em um arquivo    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plt.savefig(f'{output_path}/confusion_matrix.png', bbox_inches="tight")  # Ajusta os limites da imagem para caber tudo
    plt.close(fig)  # Fecha a figura para liberar memória


if __name__ == "__main__":
    model_path = r'\\storage01\Robotica\CME\dataset_cme_v4\laparoscopia_06-2024\treinamentos\treino concluido\onnx\best.onnx'
    
    with open('./dataset/tools.yaml', 'r') as file:
        dict_yaml = yaml.safe_load(file)

    classes = dict_yaml['names']
    input_size = (1088, 1088)
    # model = OpencvOnnx(model_path, classes, input_size)
    model = OnnxYolo(model_path, classes)
    validation(model, dict_yaml, output_path="./runs/val/onnx_yolo")
    
    # To export 
    # python .\export.py --data ./datasets/tools/tools.yaml --weights ./weights/best_yolo_s.pt --imgsz (1088, 1088) --include onnx                                                 
    