import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
import visualize
import time
from imutils.video import WebcamVideoStream
# load the class label names from disk, one label per line
# CLASS_NAMES = open("coco_labels.txt").read().strip().split("\n")

CLASS_NAMES = ['BG', 'blouse', 'crop-top', 'jeans', 'dress', 'jumper', 'shorts', 'skirt', 'trousers', 't-shirt']

class SimpleConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "fashion_inference"
    
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)

# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=SimpleConfig(),
                             model_dir=os.getcwd())

# Load the weights into the model.
model.load_weights(filepath="Fashion_mask_rcnn_trained.h5",by_name=True)
#model.load_weights(filepath="C:\\Users\\phfro\\PycharmProjects\\Mask-RCNN\\kangaroo-transfer-learning\\Kangaro_mask_rcnn_trained.h5",by_name=True)

# load the input image, convert it from BGR to RGB channel
#frame = cv2.imread("C:\\Users\\phfro\\PycharmProjects\Mask-RCNN\\fashion-transfer-learning\\fashion\\test.jpg")
#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Open webcam at the ID 0
#cap = cv2.VideoCapture(0)#Check whether user selected camera is opened successfully.
#if not (cap.isOpened()):
#    print('camera not working')

vs = cv2.VideoCapture(0)

#cv2.namedWindow(SCREEN_NAME, cv2.WINDOW_NORMAL)
#cv2.setWindowProperty(SCREEN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# Capture frame-by-frame

while True:

#    if OPTIMIZE_CAM:
#frame = vs.read()
#    else:
    grabbed, frame = vs.read()

    #frame = cv2.resize(frame, (640, 480))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#    frame = cv2.resize(frame, (640, 480))
#    if not grabbed:
#        break

#    if SHOW_FPS_WO_COUNTER:
#        start_time = time.time()  # start time of the loop

#if PROCESS_IMG:

    results = model.detect([frame])
    r = results[0]
# Run detection
#masked_image = mrcnn.visualize.display_instances(frame, r['rois'],
#      r['masks'],r['class_ids'], CLASS_NAMES, r['scores'])
    masked_image = mrcnn.visualize.get_masked_image(frame, r['rois'],
      r['masks'],r['class_ids'], CLASS_NAMES, r['scores'])

#if PROCESS_IMG:
    if not masked_image is None:
        s = masked_image
    else:
        s = frame


# Display the frame
    s = cv2.resize(s, (800, 600))
    cv2.imshow("test", s)
    if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# When everything is done, release the camera from video capture
#if OPTIMIZE_CAM:
#    vs.stop()
#else:
#vs.release()
#cv2.destroyAllWindows()




# Perform a forward pass of the network to obtain the results
#r = model.detect([image])

# Get the results for the first image.
#r = r[0]

# Visualize the detected objects.
#mrcnn.visualize.display_instances(image=image,
#                                  boxes=r['rois'],
#                                  masks=r['masks'],
#                                  class_ids=r['class_ids'],
#                                  class_names=CLASS_NAMES,
#                                  scores=r['scores'])
