import cv2
import landmark_extraction as pointsCalculator
import nail_detector as nd
import hand_detector


def start(testdata_path, enable_hand_detection=False, framesToProcess=100, displayHandDetection=False, displayNailDetection=False, displayLandmarkExtraction=False):
    frames = []

    if enable_hand_detection:
        frames = hand_detector.detect_hands_create_boundingbox(
            testdata_path, displayHandDetection)
    else:
        cap = cv2.VideoCapture(testdata_path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ret, frame = cap.read()
        while ret:
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            frames.append(frame)
            ret, frame = cap.read()

    # if 1 - Dorsal
    # if 0 - Palm
    hand_side = nd.palm_dorsal_identifier(testdata_path, displayNailDetection)

    return pointsCalculator.calculate_points(frames[:framesToProcess], hand_side, displayLandmarkExtraction)
