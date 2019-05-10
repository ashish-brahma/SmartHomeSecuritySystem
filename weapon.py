#USAGE
# python weapon.py
import cloudinary
from cloudinary.uploader import upload
from cloudinary.api import delete_resources_by_tag, resources_by_tag
from cloudinary.utils import cloudinary_url

from imutils.video import VideoStream
from imutils.video import FPS
import imutils

# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import cv2
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

from twilio.rest import Client

# Account Sid and Auth Token from twilio.com/console
account_sid = 'AC7bc49744715cf967453577356351a027'
auth_token = '097a78944fa4fcd7132727131b26193c'
client = Client(account_sid, auth_token)

cloudinary.config(
  cloud_name = 'diabloash',  
  api_key = '514458418544597',  
  api_secret = '0-AwrFfVWoILjPmiL-cDLgPxXDQ'  
)

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = 'resnet50_coco_best_v2.1.0.h5'

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
#model = models.convert_model(model)

#print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 
9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 
17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 
26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 
35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork',
 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 
 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 
 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 
 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 
 78: 'hair drier', 79: 'toothbrush'}


# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start() 
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()

while True:
    frame = vs.read()
    
    draw = frame.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    frame = preprocess_image(frame)
    frame, scale = resize_image(frame)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(frame, axis=0))
    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break
        
        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)
    
        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)
        
        if label == 43 or label == 76:
            draw = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
            cv2.imwrite("out.jpg",draw)
            cloudinary.uploader.upload("out.jpg", public_id="intruder", tags="uploaded")
            cloudinary.utils.cloudinary_url("intruder.jpg")
            message = client.messages.create(
            	           media_url = 'http://res.cloudinary.com/diabloash/image/upload/intruder.jpg',
                           body= 'Intruder Detected',
                           from_='whatsapp:+14155238886',
                           to='whatsapp:+917601897265'
                          )
            delete_resources_by_tag("uploaded")
            draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
            
    draw = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)                
    cv2.imshow("Frame", draw)
    
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

# stop the FPS counter
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

