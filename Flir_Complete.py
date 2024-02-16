import cv2
import numpy as np
import PySpin
import sys
import emoji

def load_model(model_path='last(m).onnx'):
    return cv2.dnn.readNetFromONNX(model_path)

def load_classes(file_path='sona.names'):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

def perform_detection(frame, net, classes, confidence_threshold=0.5, nms_threshold=0.5):
    #height, width, channels = frame.shape if len(frame.shape) == 3 else frame.shape + (1,)
    if len(frame.shape) == 2:  # Grayscale image
        height, width = frame.shape
        channels = 1

    # Convert grayscale to RGB by duplicating the single channel
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    #channels = 10
    else:  # Color image
        height, width, channels = frame.shape


    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=(640, 640), mean=[0, 0, 0], swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()[0]

    class_ids = []
    confidences = []
    boxes = []

    x_scale, y_scale = width / 640, height / 640

    for row in detections:
        confidence = row[4]
        if confidence > confidence_threshold:
            classes_score = row[5:]
            class_id = np.argmax(classes_score)
            if classes_score[class_id] > confidence_threshold:
                class_ids.append(class_id)
                confidences.append(confidence)
                cx, cy, w, h = row[:4]
                x1 = int((cx - w/2) * x_scale)
                y1 = int((cy - h/2) * y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)
                box = np.array([x1, y1, width, height])
                boxes.append(box)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    return [(boxes[i], classes[class_ids[i]], confidences[i]) for i in indices]

def draw_boxes(frame, detections):
    lab_list = []
    dent_count = 0
    spot_count = 0

    if len(frame.shape) == 3:
        height, width, _ = frame.shape
    else:
        height, width = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    top_middle_position = (width // 2 - 120, 40)

    for box, label, confidence in detections:
        x1, y1, w, h = box
        text = f"{label}: {confidence:.2f}"

        if label == 'Dent':
            lab_list.append(label)
            dent_count += 1
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)
            frame = cv2.putText(frame, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 20), 2, cv2.LINE_AA)
        else:
            lab_list.append(label)
            spot_count += 1
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (255, 0, 50), 2)
            frame = cv2.putText(frame, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)

    if len(lab_list) == 0:
        frame = cv2.putText(frame, "Result OK", top_middle_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        frame = cv2.putText(frame, "Result NotOK", top_middle_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        count_text = f"Dent Count: {dent_count} | Spot Count: {spot_count}"
        frame = cv2.putText(frame, count_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def main():
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()

    if cam_list.GetSize() == 0:
        print("No FLIR cameras found.")
        system.ReleaseInstance()
        return

    cam = cam_list.GetByIndex(0)
    cam.Init()
    cam.BeginAcquisition()

    net = load_model()
    classes = load_classes()

    while True:
        try:
            image_result = cam.GetNextImage(1000)
            if image_result.IsIncomplete():
                print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
                continue

            frame = image_result.GetNDArray()
            frame = cv2.resize(frame, (1000, 600))
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            detections = perform_detection(frame, net, classes)
            frame = draw_boxes(frame, detections)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("FLIR Camera", frame)

            k = cv2.waitKey(20)
            if k == ord('q'):
                print("Successfully completed")
                print(emoji.emojize(":grinning_face_with_big_eyes:"))
                
                break

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            break

    cam.EndAcquisition()
    cam.DeInit()
    cam_list.Clear()
    #system.ReleaseInstance()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
