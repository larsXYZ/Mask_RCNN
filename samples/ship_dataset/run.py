import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class ShipConfig(Config):
    """Configuration for training on the toy ships dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "ships"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

config = ShipConfig()

class ShipDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of crudely generated images of two boats
    The images are generated on the fly. No file access required.
    """

    def load_ships(self, count, height, width):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("ships", 1, "cruiseship")
        self.add_class("ships", 2, "tanker")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in range(count):
            bg_color, ships = self.random_image(height, width)
            self.add_image("ships", image_id=i, path=None,
                           width=width, height=height,
                           bg_color=bg_color, ships=ships)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
        image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)
        for ship, dims in info['ships']:
            image = self.draw_ship(image, ship, dims)
        return image

    def image_reference(self, image_id):
        """Return the ships data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "ships":
            return info["ships"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for ships of the given image ID.
        """
        info = self.image_info[image_id]
        ships = info['ships']
        count = len(ships)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (ship, dims) in enumerate(info['ships']):
            mask[:, :, i:i + 1] = self.draw_mask(mask[:, :, i:i + 1].copy(),
                                                  ship, dims)
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(
                occlusion, np.logical_not(mask[:, :, i]))
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in ships])
        return mask, class_ids.astype(np.int32)

    def rotateImage(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def image_resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        # return the resized image
        return resized

    def transform(self, ship, dims):

        x, y, s, r, m = dims

        #Scale
        if np.shape(ship[0]) > np.shape(ship[1]):
            ship = self.image_resize(ship, height=2*s)
        else:
            ship = self.image_resize(ship, width=2*s)
        #Rotate
        ship = self.rotateImage(ship,r)
        #Mirror
        if m: ship = cv2.flip(ship,1)

        return ship

    def draw_ship(self, image, ship, dims):
        """Draws a ship from the given specs."""
        # Get the center x, y and the size s
        x, y, s, r, m = dims

        #Load files
        if ship == 'cruiseship':
            im_path = 'ships/cruiseship_isolated.png'
            ma_path = 'ships/cruiseship_isolated_mask.png'
        elif ship == 'tanker':
            im_path = 'ships/tanker_isolated.png'
            ma_path = 'ships/tanker_isolated_mask.png'

        #Transforming ship
        ship = cv2.imread(im_path)
        ship_transformed = self.transform(ship.copy(), dims)
        ship_shape = np.shape(ship_transformed)
        s_x = int((ship_shape[0]+0.5)//2)
        s_y = int((ship_shape[1]+0.5)//2)
        ship_transformed = ship_transformed[0:np.shape(image[x-s_x:x+s_x, y-s_y:y+s_y, :])[0],
                                            0:np.shape(image[x-s_x:x+s_x, y-s_y:y+s_y, :])[1],
                                            :]
        ship_transformed_th = self.threshold(ship_transformed)

        #Adding boat to image
        image_slice = image[x - s_x:x + s_x, y - s_y:y + s_y, :]
        image_slice -= 255*image_slice*ship_transformed_th
        image_slice += ship_transformed
        image[x - s_x:x + s_x, y - s_y:y + s_y, :] = image_slice

        return image

    def threshold(self, image):
        image_gs = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        image_gs = np.expand_dims(image_gs, 2)

        _, image_th = cv2.threshold(image_gs, 10, 255, cv2.THRESH_BINARY)
        image_th = np.expand_dims(image_th, 2)

        return image_th

    def draw_mask(self, image, ship, dims):
        """Draws a mask from the given specs."""
        # Get the center x, y and the size s
        x, y, s, r, m = dims

        #Load files
        if ship == 'cruiseship':
            ma_path = 'ships/cruiseship_isolated_mask.png'
            id = 1
        elif ship == 'tanker':
            ma_path = 'ships/tanker_isolated_mask.png'
            id = 2

        #Transforming mask
        mask = cv2.imread(ma_path)
        mask_transformed = self.transform(mask.copy(), dims)
        mask_shape = np.shape(mask_transformed)
        s_x = int((mask_shape[0]+0.5)//2)
        s_y = int((mask_shape[1]+0.5)//2)
        mask_transformed = mask_transformed[0:np.shape(image[x-s_x:x+s_x, y-s_y:y+s_y, :])[0],
                                            0:np.shape(image[x-s_x:x+s_x, y-s_y:y+s_y, :])[1],
                                            :]

        mask_transformed_th = self.threshold(mask_transformed)

        #Adding mask to image
        image[x-s_x:x+s_x, y-s_y:y+s_y, :] = id/255*mask_transformed_th

        return image

    def random_ship(self, height, width):
        """Generates specifications of a random shape that lies within
        the given height and width boundaries.
        Returns a tuple of three valus:
        * The shape name (square, circle, ...)
        * Shape color: a tuple of 3 values, RGB.
        * Shape dimensions: A tuple of values that define the shape size
                            and location. Differs per shape type.
        """
        # Ship
        ship = random.choice(["cruiseship", "tanker"])
        # Center x, y
        buffer = 20
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
        # Size
        s = random.randint(buffer, height // 8)
        if s % 2 == 1: s -= 1
        # Rotation
        r = random.randint(-5,5)
        # Mirror
        m = random.randint(0,1)
        return ship, (x, y, s, r, m)

    def random_image(self, height, width):
        """Creates random specifications of an image with multiple ships.
        Returns the background color of the image and a list of shape
        specifications that can be used to draw the image.
        """
        # Pick random background color (blueish)
        bg_color = np.array([random.randint(200, 255) for _ in range(3)])
        bg_color[0] = bg_color[0] / 4
        bg_color[1] = bg_color[1] / 4
        # Generate a few random ships and record their
        # bounding boxes
        ships = []
        boxes = []
        N = random.randint(1, 4)
        for _ in range(N):
            shape, dims = self.random_ship(height, width)
            ships.append((shape, dims))
            x, y, s, _, _ = dims
            boxes.append([y - s, x - s, y + s, x + s])
        # Apply non-max suppression wit 0.3 threshold to avoid
        # ships covering each other
        keep_ixs = utils.non_max_suppression(
            np.array(boxes), np.arange(N), 0.3)
        ships = [s for i, s in enumerate(ships) if i in keep_ixs]
        return bg_color, ships


# Training dataset
dataset_train = ShipDataset()
dataset_train.load_ships(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 5)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    plt.imshow(image)
    print(np.shape(mask))