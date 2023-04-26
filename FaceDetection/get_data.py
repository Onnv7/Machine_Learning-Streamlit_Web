import argparse
import shutil
import os
import numpy as np
import cv2 as cv2
import glob


def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError


parser = argparse.ArgumentParser()
parser.add_argument('--image1', '-i1', type=str,
                    help='Path to the input image1. Omit for detecting on default camera.')
parser.add_argument('--image2', '-i2', type=str,
                    help='Path to the input image2. When image1 and image2 parameters given then the program try to find a face on both images and runs face recognition algorithm.')
parser.add_argument('--video', '-v', type=str, help='Path to the input video.')
parser.add_argument('--scale', '-sc', type=float, default=1.0,
                    help='Scale factor used to resize input video frames.')
parser.add_argument('--face_detection_model', '-fd', type=str, default='./FaceDetection/model/face_detection_yunet_2022mar.onnx',
                    help='Path to the face detection model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet')
parser.add_argument('--face_recognition_model', '-fr', type=str, default='./FaceDetection/model/face_recognition_sface_2021dec.onnx',
                    help='Path to the face recognition model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface')
parser.add_argument('--score_threshold', type=float, default=0.9,
                    help='Filtering out faces of score < score_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3,
                    help='Suppress bounding boxes of iou >= nms_threshold.')
parser.add_argument('--top_k', type=int, default=5000,
                    help='Keep top_k bounding boxes before NMS.')
parser.add_argument('--save', '-s', type=str2bool, default=False,
                    help='Set true to save results. This flag is invalid when using camera.')
args = parser.parse_args()


def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(
                idx, face[0], face[1], face[2], face[3], face[-1]))

            coords = face[:-1].astype(np.int32)
            cv2.rectangle(input, (coords[0], coords[1]), (coords[0] +
                                                          coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv2.circle(input, (coords[4], coords[5]),
                       2, (255, 0, 0), thickness)
            cv2.circle(input, (coords[6], coords[7]),
                       2, (0, 0, 255), thickness)
            cv2.circle(input, (coords[8], coords[9]),
                       2, (0, 255, 0), thickness)
            cv2.circle(input, (coords[10], coords[11]),
                       2, (255, 0, 255), thickness)
            cv2.circle(input, (coords[12], coords[13]),
                       2, (0, 255, 255), thickness)
    cv2.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


detector = cv2.FaceDetectorYN.create(
    args.face_detection_model,
    "",
    (320, 320),
    args.score_threshold,
    args.nms_threshold,
    args.top_k
)
recognizer = cv2.FaceRecognizerSF.create(
    args.face_recognition_model, "")

tm = cv2.TickMeter()

cap = cv2.VideoCapture(0)
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
detector.setInputSize([frameWidth, frameHeight])


def create_data(name, index, frame):
    # Inference
    tm.start()
    faces = detector.detect(frame)  # faces is a tuple
    tm.stop()
    flag = True
    # if key == ord('s') or key == ord('S'):
    if faces[1] is not None:
        face_align = recognizer.alignCrop(frame, faces[1][0])
        file_name = './FaceDetection/data/'+name+'/' + name+'%04d.bmp' % index
        cv2.imwrite(file_name, face_align)
    if faces[1] is None:
        flag = False
    # Draw results on the input image
    visualize(frame, faces, tm.getFPS())

    # Visualize results

    return frame, flag


def clear_file_content(file_path):
    with open(file_path, 'w') as file:
        file.truncate(0)


def clear_data(folder_name):
    folder_path = "./FaceDetection/data/" + folder_name

    files = os.listdir(folder_path)

    for file in files:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


def remove_folder(folder_name):
    folder_path = "./FaceDetection/data/" + folder_name
    shutil.rmtree(folder_path)


def get_id_name_list(folder_path):

    folder_names = []
    id = []
    for folder_name in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, folder_name)):
            parts = folder_name.split(' - ')
            id.append(parts[0])
            folder_names.append(parts[1])

    return id, folder_names


def get_id_name_quantity_list(folder_path):

    folder_names = []
    id = []
    quantity = []
    for folder_name in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, folder_name)):
            parts = folder_name.split(' - ')
            id.append(parts[0])
            folder_names.append(parts[1])
            files = glob.glob(os.path.join(folder_path, folder_name, '*'))
            quantity.append(len(files))

    return id, folder_names, quantity


def get_folder_name(folder_path):
    folder_names = []
    for folder_name in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, folder_name)):
            folder_names.append(folder_name)
    return folder_names[1:]


def make_folder(id, folder_name):
    id = "{:06d}".format(int(id))
    folder_path = "./FaceDetection/data/" + id + " - " + folder_name

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        return True
    else:
        return False


if __name__ == '__main__':
    # main()
    pass
