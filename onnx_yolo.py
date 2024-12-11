import onnxruntime as ort
import numpy as np
import cv2

class OnnxYolo():
    def __init__(self, model_path: str, classes: dict):
        # Load model
        self.session = ort.InferenceSession(model_path)

        # Classes and colors
        self.classes = classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # Input datas
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.input_height, self.input_width = input_shape[2], input_shape[3]

        # Output data
        self.output_name = self.session.get_outputs()[0].name
    
    def detect(self, image, confidence=0.3, iou=0.5, show=False):
        image_shape = image.shape
        input_data = self._preprocess(image)

        # Make inference
        output_name = self.session.get_outputs()[0].name
        outputs = self.session.run([output_name], {self.input_name: input_data})

        # Post process inference
        result_boxes, scores, boxes, class_ids = self._postprocess(outputs, image_shape, confidence_thres=confidence, iou_thres=iou)
        detections = self._make_dicitionary_detections(image_shape, result_boxes, scores, boxes, class_ids)

        if show:
             self._draw_detections(image, boxes, scores, class_ids)

        return detections


    def _preprocess(self, image):
            """
            Preprocesses the input image before performing inference.

            Returns:
                image_data: Preprocessed image data ready for inference.
            """
            # Read the input image using OpenCV

            # Get the height and width of the input image
            
            img = image.copy()

            # Convert the image color space from BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize the image to match the input shape
            img = cv2.resize(img, (self.input_width, self.input_height))

            # Normalize the image data by dividing it by 255.0
            image_data = np.array(img) / 255.0

            # Transpose the image to have the channel dimension as the first dimension
            image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

            # Expand the dimensions of the image data to match the expected input shape
            image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

            # Return the preprocessed image data
            return image_data

    def _postprocess(self, output, image_shape, confidence_thres = 0.5, iou_thres = 0.5):
            """
            Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

            Args:
                input_image (numpy.ndarray): The input image.
                output (numpy.ndarray): The output of the model.

            Returns:
                numpy.ndarray: The input image with detections drawn on it.
            """
            # Transpose and squeeze the output to match the expected shape
            outputs = np.transpose(np.squeeze(output[0]))

            # Get the number of rows in the outputs array
            rows = outputs.shape[0]

            # Lists to store the bounding boxes, scores, and class IDs of the detections
            boxes = []
            scores = []
            class_ids = []

            img_height, img_width, _ = image_shape

            # Calculate the scaling factors for the bounding box coordinates
            x_factor = img_width / self.input_width
            y_factor = img_height / self.input_height

            # Iterate over each row in the outputs array
            for i in range(rows):
                # Extract the class scores from the current row
                classes_scores = outputs[i][4:]

                # Find the maximum score among the class scores
                max_score = np.amax(classes_scores)

                # If the maximum score is above the confidence threshold
                if max_score >= confidence_thres:
                    # Get the class ID with the highest score
                    class_id = np.argmax(classes_scores)

                    # Extract the bounding box coordinates from the current row
                    x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                    # Calculate the scaled coordinates of the bounding box
                    x_min = int((x - w / 2) * x_factor)
                    y_min = int((y - h / 2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    # Add the class ID, score, and box coordinates to the respective lists
                    class_ids.append(class_id)
                    scores.append(max_score)
                    boxes.append([x_min, y_min, width, height])

            # Apply non-maximum suppression to filter out overlapping bounding boxes
            result_boxes = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)

            # Return the modified input image
            return result_boxes, scores, boxes, class_ids

    def _make_dicitionary_detections(self, image_shape, result_boxes, scores, bboxes, class_ids):
            detections = []
            height, width, _ = image_shape

            # Iterate through NMS results to draw bounding boxes and labels
            for result in result_boxes:
                bbox = bboxes[result]
                x_min = bbox[0] / width
                y_min = bbox[1] / height
                w = bbox[2] / width
                h = bbox[3] / height
                detection = {
                    "class_id": int(class_ids[result]),
                    "class_name": self.classes[class_ids[result]],
                    "confidence": scores[result],
                    "bbox_n": [x_min, y_min, w, h],
                }
                detections.append(detection)
            
            return detections

    # Exibir resultados em uma imagem
    def _draw_detections(self, image, boxes, scores, class_ids):
            """
            Draws bounding boxes and labels on the input image based on the detected objects.

            Args:
                img: The input image to draw detections on.
                box: Detected bounding box.
                score: Corresponding detection score.
                class_id: Class ID for the detected object.

            Returns:
                None
            """
            img = image.copy()
            img = cv2.resize(img, (1000, 750))
            hr, wr, _ = img.shape
            ho, wo, _ = image.shape
            x_factor = wr / wo
            y_factor = hr / ho
            
            for i, box in enumerate(boxes):
                # Extract the coordinates of the bounding box
                x1, y1, w, h = box
                x1 = int(x1 * x_factor)
                y1 = int(y1 * y_factor)
                w = int(w * x_factor)
                h = int(h * y_factor)

                # Retrieve the color for the class ID
                color = self.color_palette[class_ids[i]]

                # Draw the bounding box on the image
                cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

                # Create the label text with class name and score
                label = f"{self.classes[class_ids[i]]}: {scores[i]:.2f}"

                # Calculate the dimensions of the label text
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                # Calculate the position of the label text
                label_x = x1
                label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

                # Draw a filled rectangle as the background for the label text
                cv2.rectangle(
                    img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
                )

                # Draw the label text on the image
                cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
            # Display the output image in a window
            cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
            cv2.imshow("Output", img)

            # Wait for a key press to exit
            cv2.waitKey(0)

if __name__ == '__main__':
    model_path = r"\\storage01\Robotica\CME\dataset_cme_v4\laparoscopia_06-2024\treinamentos\treino concluido\onnx\best.onnx"

    # Classes do modelo YOLOv9
    classes = {
        0: 'meryland',
        1: 'aspirador 5mm',
        2: 'hook in L',
        3: 'debakey',
        4: 'grasper',
        5: 'clamp',
        6: 'tesoura',
        7: 'porta agulha',
        8: 'cachorrinho'
    }

    model = OnnxYolo(model_path, classes)

    image = cv2.imread(r'\\storage01\Robotica\CME\dataset_cme_v4\laparoscopia_06-2024\f1 - meryland\images\maryland_0.jpg')
    detections = model.detect(image, show=True)

    # Realizar inferência
    print(detections)



