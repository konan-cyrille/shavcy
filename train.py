from Classification.Custom import TrainingModel

mdl_trainer = TrainingModel()
mdl_trainer.setModelAsResnet()
mdl_trainer.setDataDirectory("datasplit")
mdl_trainer.split_Data_folder(folder_to_split="../GardenInSky/plantsImg", ratio=(.8, .1, .1), seed=89)