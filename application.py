# filter warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# keras imports
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input

# other imports
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
import cv2
import h5py
import os
import json
import pickle as cPickle
import datetime
import time
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

lab = ["Acne_and_Rosacea","Atopic Dermatitis","Bullous_Disease","Cellulitis_Impetigo_and_other_Bacterial_Infections","Exanthems_and_Drug_Eruptions","Hair_Loss_Alopecia_and_other_Hair_Diseases","Herpes_HPV_and_other_STDs","Light_Diseases_and_Disorders_of_pigmentation","Lupus_and_other_Connective_Tissue_diseases","Melanoma_Skin_Cancer_Nevi_and_Moles","Nail_Fungus_and_other_Nail_Disease","Poison Ivy Photos and other Contact Dermatitis", "Psoriasis pictures Lichen Planus and related diseases","Scabies Lyme Disease and other Infestations and Bites","Seborrheic Keratoses and other Benign Tumors","Systemic Disease","Urticaria Hives","Vascular Tumors","Vasculitis Photos","Warts Molluscum and other Viral Infections"]

# load the user configs
with open('conf/conf.json') as f:    
	config = json.load(f)

# config variables
model_name = config["model"]
weights = config["weights"]
include_top = config["include_top"]
train_path = config["train_path"]
features_path = config["features_path"]
labels_path = config["labels_path"]
#test_size = config["test_size"]
results = config["results"]
model_path = config["model_path"]
classifier_path = config["classifier_path"]
# create the pretrained models
# check for pretrained weight usage or not
# check for top layers to be included or not
if model_name == "vgg16":
	base_model = VGG16(weights=weights)
	model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
	image_size = (224, 224)
elif model_name == "vgg19":
	base_model = VGG19(weights=weights)
	model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
	image_size = (224, 224)
elif model_name == "resnet50":
	base_model = ResNet50(weights=weights)
	model = Model(input=base_model.input, output=base_model.get_layer('flatten').output)
	image_size = (224, 224)
elif model_name == "inception_v3":
	base_model = InceptionV3(include_top=include_top, weights=weights, input_tensor=Input(shape=(299,299,3)))
	model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
	image_size = (299, 299)
elif model_name == "inceptionresnetv2":
	base_model = InceptionResNetV2(include_top=include_top, weights=weights, input_tensor=Input(shape=(299,299,3)))
	model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
	image_size = (299, 299)
elif model_name == "mobilenet":
	base_model = MobileNet(include_top=include_top, weights=weights, input_tensor=Input(shape=(224,224,3)), input_shape=(224,224,3))
	model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
	image_size = (224, 224)
elif model_name == "xception":
	base_model = Xception(weights=weights)
	model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
	image_size = (299, 299)
else:
	base_model = None
print ("[INFO] successfully loaded base model and model...")

loaded_model = cPickle.load(open(classifier_path, 'rb'))

print ("[INFO] successfully Loaded Trained Model...")

cur_path = "test"
for test_path in glob.glob(cur_path + "/*.jpg"):
	#load = i + ".png"
	print ("[INFO] loading", test_path,"image ")
	img = image.load_img(test_path, target_size=image_size)
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	feature = model.predict(x)
	#flat = feature.flatten()
	preds = loaded_model.predict(feature)
	print (preds)
	print ("I think the disease is : ",lab[preds[0]])
	show_image = cv2.imread(test_path)
	show_image = cv2.resize(show_image, (500, 500)) 
	#disease = preds
	#print (show_image)
	cv2.putText(show_image, lab[preds[0]], (40,50), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 0), 2)
	cv2.imshow("result",show_image)
	cv2.waitKey(0)