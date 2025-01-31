#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
import time
import os
import torch
#import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from sensor_msgs.msg import Image as SensorImage
from mixed_reality.msg import Control
from cv_bridge import CvBridge

from stable_baselines3 import SAC  # Import the Stable-Baselines3 SAC
from training.version10_RL_GAN import Generator

# Path to the Stable-Baselines3 SAC model
#MODEL_FILE = 'regular_GAN_m1.zip'
MODEL_FILE = 'very_good_GAN_m2.zip'
CYCLEGAN_FILE = 'CarF_netG_AB_epoch_9.pth'

FIXED_THROTTLE = True

STEERING = 0
THROTTLE = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Adjust this path to your own environment if needed:
dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAN_path = os.path.join(dir_path, "CycleGAN", CYCLEGAN_FILE)
model_path = os.path.join(dir_path, "final_models", MODEL_FILE)

netG_AB = Generator(input_nc=3, output_nc=3, ngf=64, n_residual_blocks=6).to(device)
netG_AB.load_state_dict(torch.load(GAN_path, map_location=device))
netG_AB.eval()

# Transforms for the CycleGAN input
cyclegan_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                            [0.5, 0.5, 0.5])
])

model = None
pub_throttle_steering = None
bridge = CvBridge()

prev_time = None


def preprocess_cyclegan(observation: np.ndarray,
                        netG_AB: torch.nn.Module,
                        device: torch.device) -> np.ndarray:
    """
    Convert RGBA->RGB if needed, resize to (128,128), scale to [-1,1],
    run CycleGAN on GPU, then resize to (60,80) in Torch, 
    and return a (3,60,80) NumPy array in [0,1].
    """

    t1 = time.time()

    # 1) Drop alpha if present
    if observation.shape[2] == 4:  
        observation = observation[:, :, :3]

    # 2) Resize (H,W,3) from e.g. (240,320) -> (128,128)
    obs_resized = cv2.resize(observation, (128, 128), interpolation=cv2.INTER_AREA)

    # 3) Convert [0,255] -> [0,1], then [0,1] -> [-1,1]
    obs_resized = obs_resized.astype(np.float32) / 255.0
    obs_resized = (obs_resized - 0.5) / 0.5  # now in [-1,1]
    # shape is (128,128,3)

    # 4) Transpose to (3,128,128), make a batch dimension, move to GPU
    obs_torch = torch.from_numpy(obs_resized.transpose(2,0,1)).unsqueeze(0).to(device)
    # shape: (1,3,128,128)

    with torch.no_grad():
        # 5) Run CycleGAN forward pass, output in [-1,1]
        fake_car = netG_AB(obs_torch)  # shape: (1,3,128,128)

        # 6) Convert [-1,1] -> [0,1]
        fake_car = (fake_car * 0.5) + 0.5

        # 7) Resize from (128,128) -> (60,80) in Torch (GPU)
        fake_car = F.interpolate(fake_car, size=(60, 80), mode='bilinear', align_corners=False)
        # shape: (1,3,60,80)

    # Move to CPU and convert to NumPy float32
    fake_car_np = fake_car.squeeze(0).cpu().numpy()
    # shape: (3,60,80) in [0,1]

    t2 = time.time()    
    print(f"Preprocess time: {t2-t1}")

    return fake_car_np

def preprocess_image(observation: np.ndarray) -> np.ndarray:
    """
    Runs CycleGAN on the raw observation, then resizes to (80, 60), normalizes to [0,1],
    and returns a (C, H, W) float32 array.
    """
    global netG_AB
    global cyclegan_transform
    global device

    #t1 = time.time()

    # Convert RGBA to RGB if needed
    if observation.shape[2] == 4:
        observation = cv2.cvtColor(observation, cv2.COLOR_RGBA2RGB)

    # Convert to PIL, apply CycleGAN transformation
    input_img = Image.fromarray(observation, "RGB")
    input_tensor = cyclegan_transform(input_img).unsqueeze(0).to(device)

    with torch.no_grad():
        fake_car = netG_AB(input_tensor)  # shape: (1,3,H,W), in [-1,1]

    # Convert from [-1,1] to [0,1]
    fake_car = (fake_car * 0.5) + 0.5
    fake_car = fake_car.squeeze(0).cpu().clamp_(0, 1)
    out_pil = transforms.ToPILImage()(fake_car)

    # Resize to (80,60)
    out_pil = out_pil.resize((80, 60), Image.Resampling.LANCZOS)
    obs_np = np.asarray(out_pil, dtype=np.float32) / 255.0  # in [0,1]
    obs_np = np.transpose(obs_np, (2, 0, 1))  # (C, H, W)

    #t2 = time.time()    
    #print(f"Preprocess time: {t2-t1}")

    return obs_np

def new_image(msg):

    global prev_time

    now = time.time()
    if prev_time is not None and (now - prev_time) < 0.0:
        return
    prev_time = now
    
    global FIXED_THROTTLE
    global model
    global pub_throttle_steering
    global bridge

    # Convert ROS image to OpenCV image
    image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')  
    #image = image[None, ...]
    image = preprocess_image(image)
    #image = preprocess_cyclegan(image, netG_AB, device)

    # Predict actions using the SAC model
    action, _ = model.predict(image, deterministic=True)
    #print(f"Action: {action}")
    steering = action[0]

    if FIXED_THROTTLE:
        throttle = 0.35

    # Publish throttle and steering commands
    if pub_throttle_steering is None:
        pub_throttle_steering = rospy.Publisher("model/throttle_steering", Control, queue_size=10)
    msg = Control()
    msg.throttle = throttle
    msg.steering = steering
    msg.brake = False
    msg.reverse = False
    msg.stopping = False
    #msg.header.stamp = rospy.Time.now()
    
    pub_throttle_steering.publish(msg)

def model_node():
    global model_path
    global model

    print(f"Loading model: {model_path}")
    model = SAC.load(model_path, device="cpu")  # Load the Stable-Baselines3 SAC model
    print("Model loaded successfully.")

    rospy.init_node("model_node", anonymous=True)
    #rate = rospy.Rate(20)  # Set the desired frequency to 3 Hz
    rospy.Subscriber("/sim/image", SensorImage, new_image)
    rospy.spin()
    #rospy.Subscriber("/camera", SensorImage, new_image)

    #while not rospy.is_shutdown():
        # Let ROS handle the callbacks and maintain a 3 Hz loop
    #    rate.sleep()

if __name__ == '__main__':
    try:
        model_node()
    except rospy.ROSInterruptException:
        pass
