import subsystem_1
import subsystem_2

if __name__ == '__main__':
    testdata_path = "input/videos/576/three_fingers_dorsal.webm"
    output_file_name = "label.txt"

    landmarks_file = subsystem_1.start(testdata_path, enable_hand_detection=True,
                                       framesToProcess=50, displayHandDetection=False, 
                                       displayNailDetection=True, 
                                       displayLandmarkExtraction=False)

    subsystem_2.start(landmarks_file, output_file_name)
