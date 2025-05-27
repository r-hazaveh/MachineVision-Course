
import cv2
import dlib
import numpy as np

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def face_swap(image_path, model_path="shape_predictor_68_face_landmarks.dat"):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)

    faces = detector(gray)
    if len(faces) < 2:
        raise Exception("Less than two faces detected.")

    landmarks = []
    for face in faces[:2]:
        shape = predictor(gray, face)
        shape_np = shape_to_np(shape)
        landmarks.append(shape_np)

    hull1 = cv2.convexHull(landmarks[0])
    hull2 = cv2.convexHull(landmarks[1])

    mask = np.zeros_like(image)
    cv2.fillConvexPoly(mask, hull1, (255, 255, 255))

    r2 = cv2.boundingRect(hull2)
    center = (r2[0] + r2[2] // 2, r2[1] + r2[3] // 2)

    output = cv2.seamlessClone(image, image, mask, center, cv2.NORMAL_CLONE)
    cv2.imwrite("output.jpg", output)
    print("Face swapped and saved as output.jpg")

if __name__ == "__main__":
    face_swap("input.jpg")
