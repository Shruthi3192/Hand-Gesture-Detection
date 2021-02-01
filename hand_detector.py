import datetime
from utils import detector_utils as detector_utils
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def detect_hands_create_boundingbox(input_path, display_frames=False):
    detection_graph, sess = detector_utils.load_inference_graph()

    score_thresh = 0.2
    num_workers = 4
    queue_size = 5

    cap = cv2.VideoCapture(input_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track
    num_hands_detect = 1

    #cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)
    ret, image_np = cap.read()
    processed_frames = []

    while ret:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, image_np = cap.read()
        # image_np = cv2.flip(image_np, 1)
        # if len(image_np) != 0:
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            continue

        # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
        # while scores contains the confidence for each of these boxes.
        # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

        boxes, scores = detector_utils.detect_objects(image_np,
                                                      detection_graph, sess)

        # draw bounding boxes on frame
        detector_utils.draw_box_on_image(num_hands_detect, score_thresh,
                                         scores, boxes, im_width, im_height,
                                         image_np)

        # Calculate Frames per second (FPS)
        num_frames += 1
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = num_frames / elapsed_time

        if (display_frames):
            # Display FPS on frame
            if (fps > 0):
                detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                                 image_np)

            cv2.imshow('Single-Threaded Detection',
                       cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            print("Hand Detector: frames processed: ", num_frames, "elapsed time: ",
                  elapsed_time, "fps: ", str(int(fps)))
        processed_frames.append(image_np)
    cv2.destroyAllWindows()
    return processed_frames
