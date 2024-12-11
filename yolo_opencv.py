import cv2.dnn
import numpy as np
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import os

class OpencvOnnx():
    def __init__(self, onnx_path: str):
        self.model = cv2.dnn.readNetFromONNX(onnx_path)

    def detect(self, opencv_image, classes: dict, confidence=0.3, show=False):
        # Make a blob image and take scale
        size_inference=(1088, 1088)
        blob, scale = self._preprocess_image(opencv_image, size=size_inference)

        # Load blob input and perform inference
        self.model.setInput(blob)
        outputs = self.model.forward()

        # Post process result
        result_boxes, scores, bboxes, class_ids = self._postprocess_outputs(outputs)

        # Turn results in dictionary
        detections = self._make_dicitionary_detections(
            result_boxes, 
            scores,
            bboxes,
            classes,
            class_ids,
            size_inference,
            scale,
            confidence
        )
        
        if show:
            self._show_detections(opencv_image, detections, classes)

        return detections
    
    def _preprocess_image(self, original_image, size):
         # Read the input image
        original_image = cv2.resize(original_image, size)
        [height, width, _] = original_image.shape

        # Prepare a square image for inference
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image

        # Calculate scale factor
        scale = length / size[0]

        # Preprocess the image and prepare blob for model
        blob = cv2.dnn.blobFromImage(image, scalefactor=(1/255), size=size, swapRB=True)

        return blob, scale
    
    def _postprocess_outputs(self, outputs):
         # Prepare output array
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        bboxes = []
        scores = []
        class_ids = []

        # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.25:
                bbox = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2],
                    outputs[0][i][3],
                ]
                bboxes.append(bbox)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        # Apply NMS (Non-maximum suppression)
        result_boxes = cv2.dnn.NMSBoxes(bboxes, scores, 0.50, 0.45, 0.5)

        return result_boxes, scores, bboxes, class_ids

    def _make_dicitionary_detections(self, 
                                     result_boxes, 
                                     scores, bboxes, 
                                     classes, 
                                     class_ids, 
                                     size_inference, 
                                     scale, 
                                     confidence):
        
        detections = []

        # Iterate through NMS results to draw bounding boxes and labels
        for result in result_boxes:
            if scores[result] > confidence:
                bbox = bboxes[result]
                bbox_n = np.array(bbox) / size_inference[0]
                detection = {
                    "class_id": class_ids[result],
                    "class_name": classes[class_ids[result]],
                    "confidence": scores[result],
                    "bbox": bbox,
                    "bbox_n": bbox_n,
                    "scale": scale,
                }
                detections.append(detection)
        
        return detections
    
    def _show_detections(self, image, detections, classes):
        img = image
        for detection in detections:
                    bbox = detection['bbox']
                    scale = detection['scale']

                    self._draw_bounding_box(
                        img,
                        detection['class_id'],
                        detection['confidence'],
                        round(bbox[0] * scale),
                        round(bbox[1] * scale),
                        round((bbox[0] + bbox[2]) * scale),
                        round((bbox[1] + bbox[3]) * scale),
                        classes
                    )

        # Display the image with bounding boxes
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def _draw_bounding_box(self, image, class_id, confidence, x_min, y_min, x_max, y_max, classes):
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        label = f"{classes[class_id]} ({confidence:.2f})"
        color = colors[class_id]
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(image, label, (x_min - 10, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


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

def validation(model, dict_yaml, output_path = "./runs/val/onnx"):
    classes = dict_yaml["names"]
    confusion_matrix = np.zeros((len(classes)+1,len(classes)+1))
    images = get_images_path(dict_yaml)
    
    for image_path in tqdm(images):
        image = cv2.imread( image_path)
        detections = model.detect(image, classes)
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
    with open('./dataset/tools.yaml', 'r') as file:
        dict_yaml = yaml.safe_load(file)

    model = OpencvOnnx(r'\\storage01\Robotica\CME\dataset_cme_v4\laparoscopia_06-2024\treinamentos\treino concluido\onnx\best.onnx')
    validation(model, dict_yaml)
    
    # To export 
    # python .\export.py --data ./datasets/tools/tools.yaml --weights ./weights/best_yolo_s.pt --imgsz (1088, 1088) --include onnx                                                 
    