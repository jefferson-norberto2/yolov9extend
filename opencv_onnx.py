import cv2.dnn
import numpy as np

class OpencvOnnx():
    def __init__(self, onnx_path: str, classes: dict, model_image_size: tuple[int,int]):
        self.model = cv2.dnn.readNetFromONNX(onnx_path)
        self.classes = classes
        self.input_size = model_image_size

    def detect(self, opencv_image, confidence=0.3, show=False):
        """
            Detects objects in an image using a pre-trained model.

            This method takes an OpenCV image, processes it through a neural network model, 
            and returns the detected objects as a list of dictionary. Optionally, the detections 
            can be visualized on the input image.

            Parameters:
                opencv_image (numpy.ndarray): 
                    The input image in OpenCV format (BGR) for object detection.
                
                classes (dict): 
                    A dictionary mapping class IDs to class names. Example {0: 'Car', 1: 'Person'}.

                confidence (float, optional): 
                    The confidence threshold for filtering out weak detections. 
                    Only detections with confidence scores above this threshold will be included. 
                    Defaults to 0.3.
                
                model_image_size (tuple[int, int]):
                    A tuple that define input size of model, defined by widht and heigth from image
                    in pixels.

                show (bool, optional): 
                    If True, the method will display the image with the detected objects drawn on it.
                    Defaults to False.

            Returns:
                List of dict: 
                    A list of dictionary, containing the detected objects. 
                    The structure defined is:
                    "class_id": int number of class,
                    "class_name": string name from class,
                    "confidence": float score of detection,
                    "bbox_n": bounding box containg x_min, y_min, widht and heigth normalized between 
                    0 and 1.
            """

        # Make a blob image
        blob = self._preprocess_image(opencv_image)

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
            class_ids,
            self.input_size,
            confidence
        )
        
        if show:
            self._show_detections(opencv_image, detections)

        return detections
    
    def _preprocess_image(self, original_image):
         # Read the input image
        resized_image = cv2.resize(original_image, self.input_size)
        [height, width, _] = resized_image.shape

        # Prepare a square image for inference
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = resized_image

        # Preprocess the image and prepare blob for model
        blob = cv2.dnn.blobFromImage(image, scalefactor=(1/255), size=self.input_size, swapRB=True)

        return blob
    
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
                                     class_ids, 
                                     size_inference, 
                                     confidence):
        
        detections = []

        # Iterate through NMS results to draw bounding boxes and labels
        for result in result_boxes:
            if scores[result] > confidence:
                bbox = bboxes[result]
                bbox_n = np.array(bbox) / size_inference[0]
                detection = {
                    "class_id": class_ids[result],
                    "class_name": self.classes[class_ids[result]],
                    "confidence": scores[result],
                    "bbox_n": bbox_n,
                }
                detections.append(detection)
        
        return detections
    
    def _show_detections(self, image, detections):
        img = image.copy()
        img = cv2.resize(img, (1088, 1088))
        h, w, _ = img.shape
        for detection in detections:
                    bbox = detection['bbox_n']

                    x_min = int(bbox[0] * w)
                    y_min = int(bbox[1] * h)
                    x_max = int(x_min + (bbox[2] * w))
                    y_max = int(y_min + (bbox[3] * w))

                    self._draw_bounding_box(
                        img,
                        detection['class_id'],
                        detection['confidence'],
                        x_min,
                        y_min,
                        x_max,
                        y_max,
                        self.classes
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

if __name__ == '__main__':
     model_path = r'\\storage01\Robotica\CME\dataset_cme_v4\laparoscopia_06-2024\treinamentos\treino concluido\onnx\best.onnx'
     
     # Dict with id and classes names
     classes = {0: 'meryland',
            1: 'aspirador 5mm',
            2: 'hook in L',
            3: 'debakey',
            4: 'grasper',
            5: 'clamp',
            6: 'tesoura',
            7: 'porta agulha',
            8: 'cachorrinho'}
     
     model_image_size = (1088, 1088)
     
     model = OpencvOnnx(model_path, classes, model_image_size)

     image = cv2.imread(r'\\storage01\Robotica\CME\dataset_cme_v4\laparoscopia_06-2024\f1 - meryland\images\maryland_0.jpg')
     detections = model.detect(opencv_image=image, confidence=0.3, show=True)
     print(detections)
     