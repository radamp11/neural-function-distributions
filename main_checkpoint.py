from checkpoint.checkpoint_training import CheckpointTrainer

path_to_model = '2023-05-31_22-35_shapenet_voxels_experiment'
model_idx = 3

checkpoint_trainer = CheckpointTrainer(path_to_model, model_idx)
checkpoint_trainer.continue_training()
