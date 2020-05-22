from Classification.Custom import TrainingModel
from Detection.Custom import TrainingObjectDetection

"""mdl_trainer = TrainingModel()
mdl_trainer.setModelAsResnet()
mdl_trainer.setDataDirectory("datasplit")
mdl_trainer.split_Data_folder(folder_to_split="../GardenInSky/plantsImg", ratio=(.8, .1, .1), seed=89)
mdl_trainer.train_model()"""

mdl_obj_trainer = TrainingObjectDetection()
mdl_obj_trainer.setDataDirectory("repoWork")
#mdl_obj_trainer.get_data("../GardenInSky/Detection")
#mdl_obj_trainer.split_data()
mdl_obj_trainer.train_model("detection_model-ex-010--loss-0016.345.h5", ["Tomate saine"], use_pretrained_model=True)
#mdl_obj_trainer.try_model("../GardenInSky/Detection/img_plante_test.jpg", "detection_model-ex-010--loss-0016.345.h5")
