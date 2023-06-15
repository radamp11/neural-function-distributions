from checkpoint.checkpoint_training import CheckpointTrainer

path_to_model = 'trained_models/moje/shapenet_voxels/control'
model_idx = 0

checkpoint_trainer = CheckpointTrainer(path_to_model, model_idx)
checkpoint_trainer.continue_training()
