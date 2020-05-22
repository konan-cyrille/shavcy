import os
import random
import shutil
from imageai.Detection.Custom import DetectionModelTrainer
from imageai.Detection.Custom import CustomObjectDetection
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# exécuter un code ecrit en tensorflow 1.x dans un environnement tensorflow 2.x
import tensorflow.compat.v1
tensorflow.compat.v1.disable_v2_behavior()

config = ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#config = tf.ConfigProto(allow_soft_placement=True)
#config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

class TrainingObjectDetection:

    def __init__(self):
        self.__modelType = ""
        self.__use_pretrained_model = False
        self.__data_dir = ""
        self.__data_interm_dir = ""
        self.__train_dir = ""
        self.__train_img_subdir = ""
        self.__train_annotat_subdir = ""
        self.__validation_dir = ""
        self.__validation_img_subdir = ""
        self.__validation_annotat_subdir = ""
        self.__num_epochs = 10
        self.__trained_model_dir = ""
        self.__model_config_dir = ""
        self.__initial_learning_rate = 1e-3

    def setDataDirectory(self, data_directory="", train_subdirectory="train", train_img_subdir="images",
                         train_annotat_subdir="annotations",
                         validation_subdirectory="validation", validation_img_subdir="images",
                         validation_annotat_subdir="annotations", data_interm_dir="dataInterm",
                         models_subdirectory="models", json_subdirectory="json"):

        """
        Cette methode permet initialiser la structure du repertoire de travaille

        :param data_interm_dir:
        :param validation_img_subdir:
        :param validation_annotat_subdir:
        :param train_annotat_subdir:
        :param data_directory: c'est le dossier ou se trouve les dossier train, validation et test
        :param train_subdirectory: sous dossier train, contient les images du train
        :param validation_subdirectory: sous dossier test, contient les images du test
        :param models_subdirectory: Dossier ou sera sauvegardé les models entrainé
        :param json_subdirectory: le dossier json contiendra un fichier json avec les label du model
        """
        self.__data_dir = data_directory

        self.__data_interm_dir = os.path.join(self.__data_dir, data_interm_dir)
        self.__train_dir = os.path.join(self.__data_dir, train_subdirectory)
        self.__train_img_subdir = os.path.join(self.__train_dir, train_img_subdir)
        self.__train_annotat_subdir = os.path.join(self.__train_dir, train_annotat_subdir)
        self.__validation_dir = os.path.join(self.__data_dir, validation_subdirectory)
        self.__validation_img_subdir = os.path.join(self.__validation_dir, validation_img_subdir)
        self.__validation_annotat_subdir = os.path.join(self.__validation_dir, validation_annotat_subdir)
        self.__trained_model_dir = os.path.join(self.__data_dir, models_subdirectory)
        self.__model_config_dir = os.path.join(self.__data_dir, json_subdirectory)

    def get_data(self, src_dir, sample=10):
        """
        Cette methode permet de recupérer les images d'un dossier donnée,
        de les envoyer vers un autre dossier intermidiaire,
        a partir de ce dossier intermédiaire on pourra dispacher les images dans les,
        dossiers train et validation

        :param src_dir: path du dossier ou se trouve les images à récupérer
        :param dst_dir: path dossier ou les images doivent être envoyé
        :param sample: le nombre d'image à récupérer
        """

        # Creation du repertoire de travail
        if not os.path.isdir(self.__data_dir):
            print("Initialisation du repertoire de travail ...")
            os.mkdir(self.__data_dir)
            print("fait")

        # Creation du dossier intermédiare
        if not os.path.isdir(self.__data_interm_dir):
            print("Creation du dossier intermédiare ...")
            os.mkdir(self.__data_interm_dir)
            print("fait")
        list_img_base = os.listdir(src_dir)
        subset_list_img = random.sample(list_img_base, sample)
        for img_name in subset_list_img:
            if img_name.endswith(('.jpg', '.JPG', '.png', '.PNG')):
                img_path = os.path.join(src_dir, img_name)
                dst = os.path.join(self.__data_interm_dir, img_name)
                shutil.copyfile(img_path, dst)

    def split_data(self, train_split=0.7):
        """

        :param train_split: Le nombre de valeur dans le train
        """
        # Création du dossier train et de ces sous dossier "images" et "annotations"
        if not os.path.isdir(self.__train_dir):
            os.mkdir(self.__train_dir)
            os.mkdir(self.__train_img_subdir)
            os.mkdir(self.__train_annotat_subdir)

        # Création du dossier validation et de ces sous dossier "images" et "annotations"
        if not os.path.isdir(self.__validation_dir):
            os.mkdir(self.__validation_dir)
            os.mkdir(self.__validation_img_subdir)
            os.mkdir(self.__validation_annotat_subdir)

        # Répartition des différents images dans les sous dossier
        list_img_base = os.listdir(self.__data_interm_dir)
        sample = int(len(list_img_base) * train_split)
        subset_list_img_tr = random.sample(list_img_base, sample)
        subset_list_img_val = list(set(list_img_base) - set(subset_list_img_tr))
        for img_name in subset_list_img_val:
            if not os.path.isfile(os.path.join(self.__train_img_subdir, img_name)):
                img_path = os.path.join(self.__data_interm_dir, img_name)
                dst = os.path.join(self.__validation_img_subdir, img_name)
                shutil.move(img_path, dst)

        for img_name in subset_list_img_tr:
            if not os.path.isfile(os.path.join(self.__validation_img_subdir, img_name)):
                img_path = os.path.join(self.__data_interm_dir, img_name)
                dst = os.path.join(self.__train_img_subdir, img_name)
                shutil.move(img_path, dst)

        print("les images ont été réparti dans les dossiers train/images et validation/images")

    def train_model(self, model_pretrained, label_list, batch_size=2, epoch=10, use_pretrained_model=False):

        self.__use_pretrained_model = use_pretrained_model

        if self.__use_pretrained_model:
            trainer = DetectionModelTrainer()

            # fixe le type du model comme etant yoloV3
            trainer.setModelTypeAsYOLOv3()

            trainer.setDataDirectory(self.__data_dir)
            trainer.setTrainConfig(object_names_array=label_list, batch_size=batch_size, num_experiments=epoch,
                                   train_from_pretrained_model=model_pretrained)
            trainer.trainModel()

    def try_model(self, img_to_detect_path, model_trained):
        detector = CustomObjectDetection()
        detector.setModelTypeAsYOLOv3()
        detector.setModelPath(os.path.join(self.__trained_model_dir, model_trained))
        detector.setJsonPath(os.path.join(self.__model_config_dir, "detection_config.json"))
        detector.loadModel()
        detections = detector.detectObjectsFromImage(input_image=img_to_detect_path,
                                                     output_image_path="image-detected.jpg",
                                                     minimum_percentage_probability=50, nms_treshold=0.5,)
        for detection in detections:
            print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
