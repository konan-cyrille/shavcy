import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
import split_folders
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import json

"""from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# exécuter un code ecrit en tensorflow 1.x dans un environnement tensorflow 2.x
import tensorflow.compat.v1
tensorflow.compat.v1.disable_v2_behavior()

config = ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)"""


class TrainingModel:
    #def test(self):
    #    return print("coucou je suis une methode de la classe TrainingModel")
    """
        Cette classe est une classe qui permet d'entrainer un model,
        tu defini un model de deep learning parmis les model de deep learning supporté,
        (pour l'instant un seul model est disponible Resnet50
    """

    def __init__(self):
        self.__modelType = ""
        self.__use_pretrained_model = False
        self.__data_dir = ""
        self.__train_dir = ""
        self.__validation_dir = ""
        self.__num_epochs = 10
        self.__trained_model_dir = ""
        self.__model_class_dir = ""
        self.__initial_learning_rate = 1e-3

    def setModelAsResnet(self):
        self.__modelType = ResNet50

    def setModelAsInceptionV3(self):
        self.__modelType = InceptionV3

    def setDataDirectory(self, data_directory="", train_subdirectory="train", validation_subdirectory="val",
                         models_subdirectory="models", json_subdirectory="json"):
        """
        :param data_directory: c'est le dossier ou se trouve les dossier train, validation et test
        :param train_subdirectory: sous dossier train, contient les images du train
        :param validation_subdirectory: sous dossier test, contient les images du test
        :param models_subdirectory: Dossier ou sera sauvegardé les models entrainé
        :param json_subdirectory: le dossier json contiendra un fichier json avec les label du model
        """
        self.__data_dir = data_directory

        self.__train_dir = os.path.join(self.__data_dir, train_subdirectory)
        self.__validation_dir = os.path.join(self.__data_dir, validation_subdirectory)
        self.__trained_model_dir = os.path.join(self.__data_dir, models_subdirectory)
        self.__model_class_dir = os.path.join(self.__data_dir, json_subdirectory)

    def split_Data_folder(self, folder_to_split="", ratio=(.8, .1, .1), seed=89,):
        #if os.path.isdir(self.__data_dir):
        if not os.path.isdir(self.__train_dir):
            print(os.getcwd())
            split_folders.ratio(folder_to_split, output=self.__data_dir, seed=seed, ratio=ratio)
        else:
            print("le dossier {} existe déjà".format(self.__train_dir))


    def build_model(self):

        if self.__modelType == ResNet50:

            input_shape = (224, 224, 3)
            base_resnet = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)

            x = base_resnet.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(512, activation='relu')(x)
            x = Dense(512, activation='relu')(x)
            preds = Dense(38, activation='softmax')(x)
            model = Model(inputs=base_resnet.input, outputs=preds)

            for layer in model.layers:
                layer.trainable = True

            # Optimizer
            optimizer = Adam(lr=0.0001)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                          metrics=["accuracy"])

        return model


    def train_model(self, BATCH_SIZE = 32):

        # Create_data_Splited()

        # train_path = "../dataSplited/train"
        # val_path = "../dataSplited/val"
        # test_path = "../dataSplited/test"
        # BATCH_SIZE = 32

        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           horizontal_flip=True,
                                           height_shift_range=0.1,
                                           width_shift_range=0.1)

        validation_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(directory=self.__train_dir,
                                                            target_size=(224, 224),
                                                            color_mode='rgb',
                                                            class_mode='categorical',
                                                            batch_size=BATCH_SIZE)

        validation_generator = validation_datagen.flow_from_directory(directory=self.__validation_dir,
                                                                      target_size=(224, 224),
                                                                      color_mode='rgb',
                                                                      class_mode='categorical',
                                                                      batch_size=BATCH_SIZE)
        # le nombre de fichier dans le train
        num_train = train_generator.n

        # le nombre de fichier dans la validation
        num_validation = validation_generator.n

        # Construction du model
        model = self.build_model()

        # checkpoint
        # création d'un dossier s'il n'exite pas, pour enrégistrer les checkpoints du model
        if not os.path.isdir(self.__trained_model_dir):
            os.mkdir(self.__trained_model_dir)

        if not os.path.isdir(self.__model_class_dir):
            os.mkdir(self.__model_class_dir)

        class_indices = train_generator.class_indices
        ind_class_json = {}
        for cl in class_indices:
            ind_class_json[class_indices[cl]] = cl

        with open(os.path.join(self.__model_class_dir, "model_class.json"), "w") as json_file:
            json.dump(ind_class_json, json_file, indent=4, separators=(",", " : "),
                      ensure_ascii=True)

        print("""Un document json à été crée avec le mappage des différentes classes
                    et leur indices""")

        filepath = os.path.join(self.__trained_model_dir,
                                "model-weights-epoch={epoch:02d}-val_loss={val_loss:.3f}-val_acc={val_accuracy:.3f}.h5")

        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max', period=1)

        callbacks_list = [checkpoint]

        history = model.fit(train_generator, steps_per_epoch=int(num_train / BATCH_SIZE),
                            epochs=self.__num_epochs, validation_data=validation_generator,
                            validation_steps=int(num_validation / BATCH_SIZE),
                            callbacks=callbacks_list)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        t = f.suptitle('Resnet50 Fine tune performance ', fontsize=12)
        f.subplots_adjust(top=0.85, wspace=0.3)
        epoch_list = list(range(1, 21))

        ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
        ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_xticks(np.arange(0, 26, 2))
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_title('Evolution Accuracy')
        ax1.legend(loc="best")

        ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
        ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
        ax2.set_xticks(np.arange(0, 26, 2))
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_title('Evolution Loss')
        ax2.legend(loc="best")
        graph_path = os.path.join(self.__trained_model_dir, "metrics_graph_resnet50_fine_tune.png")
        plt.savefig(graph_path)