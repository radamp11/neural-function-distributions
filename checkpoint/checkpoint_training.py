import json
import os
import sys
import time
import torch

from checkpoint.custom_training import CustomTrainer
from data.conversion import GridDataConverter, PointCloudDataConverter, ERA5Converter
from data.dataloaders import mnist, celebahq
from data.dataloaders_era5 import era5
from data.dataloaders3d import shapenet_voxels, shapenet_point_clouds
from models.discriminator import PointConvDiscriminator
from models.function_distribution import HyperNetwork, FunctionDistribution
from models.function_representation import FunctionRepresentation, FourierFeatures

from models.function_distribution import load_function_distribution


def load_function_distribution_and_optimizer(device, checkpoint, config_file=None):
    """
    """
    config, state_dict = checkpoint["config"], checkpoint["state_dict"]
    # Initialize function representation
    config_rep = config["function_representation"]
    encoding = config_rep["encoding"].to(device)
    if hasattr(encoding, 'frequency_matrix'):
        encoding.frequency_matrix = encoding.frequency_matrix.to(device)
    function_representation = FunctionRepresentation(config_rep["coordinate_dim"],
                                                     config_rep["feature_dim"],
                                                     config_rep["layer_sizes"],
                                                     encoding,
                                                     config_rep["non_linearity"],
                                                     config_rep["final_non_linearity"]).to(device)
    # Initialize hypernetwork
    config_hyp = config["hypernetwork"]
    hypernetwork = HyperNetwork(function_representation, config_hyp["latent_dim"],
                                config_hyp["layer_sizes"], config_hyp["non_linearity"]).to(device)
    # Initialize function distribution
    function_distribution = FunctionDistribution(hypernetwork).to(device)
    # Load weights of function distribution
    function_distribution.load_state_dict(state_dict)

    if config_file is not None:
        optimizer = torch.optim.Adam(
            function_distribution.hypernetwork.forward_layers.parameters(),
            lr=config_file['training']['lr'], betas=(0.5, 0.999)
        )
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        optimizer = None

    return function_distribution, optimizer


class CheckpointTrainer:

    def __init__(self, path_to_model, model_idx):
        self.path_to_model = path_to_model
        self.model_idx = model_idx

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _load_discriminator_and_optimizer_disc(self, device, checkpoint, config_file, input_dim, output_dim):

        discriminator = PointConvDiscriminator(input_dim, output_dim, config_file["discriminator"]["layer_configs"],
                                               linear_layer_sizes=config_file["discriminator"]["linear_layer_sizes"],
                                               norm_order=config_file["discriminator"]["norm_order"],
                                               add_sigmoid=True,
                                               add_batchnorm=config_file["discriminator"]["add_batchnorm"],
                                               add_weightnet_batchnorm=config_file["discriminator"][
                                                   "add_weightnet_batchnorm"],
                                               deterministic=config_file["discriminator"]["deterministic"],
                                               same_coordinates=config_file["discriminator"]["same_coordinates"]).to(
            device)

        discriminator.load_state_dict(checkpoint['state_dict'])
        # discriminator.load_state_dict(checkpoint['model'])

        optimizer_disc = torch.optim.Adam(
            discriminator.parameters(), lr=config_file['training']['lr'], betas=(0.5, 0.999)
        )

        optimizer_disc.load_state_dict(checkpoint['optimizer'])

        return discriminator, optimizer_disc

    def continue_training(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config_path = self.path_to_model + '/config.json'
        # Open config file
        with open(config_path) as f:
            config = json.load(f)

        if config["path_to_data"] == "":
            raise (
                RuntimeError("Path to data not specified. Modify path_to_data attribute in config to point to data."))

        # Create a folder to store experiment results
        timestamp = time.strftime("%Y-%m-%d_%H-%M")
        directory = "{}_{}".format(timestamp, config["id"]) + '_continue_from_idx_{}'.format(self.model_idx)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save config file in experiment directory
        with open(directory + '/config.json', 'w') as f:
            json.dump(config, f)

        # Setup dataloader
        is_voxel = False
        is_point_cloud = False
        is_era5 = False
        input_dim = None
        output_dim = None
        data_shape = None
        dataloader = None

        if config["dataset"] == 'mnist':
            dataloader = mnist(path_to_data=config["path_to_data"],
                               batch_size=config["training"]["batch_size"],
                               size=config["resolution"],
                               train=True)
            input_dim = 2
            output_dim = 1
            data_shape = (1, config["resolution"], config["resolution"])
        elif config["dataset"] == 'celebahq':
            dataloader = celebahq(path_to_data=config["path_to_data"],
                                  batch_size=config["training"]["batch_size"],
                                  size=config["resolution"])
            input_dim = 2
            output_dim = 3
            data_shape = (3, config["resolution"], config["resolution"])
        elif config["dataset"] == 'shapenet_voxels':
            dataloader = shapenet_voxels(path_to_data=config["path_to_data"],
                                         batch_size=config["training"]["batch_size"],
                                         size=config["resolution"])
            input_dim = 3
            output_dim = 1
            data_shape = (1, config["resolution"], config["resolution"], config["resolution"])
            is_voxel = True
        elif config["dataset"] == 'shapenet_point_clouds':
            dataloader = shapenet_point_clouds(path_to_data=config["path_to_data"],
                                               batch_size=config["training"]["batch_size"])
            input_dim = 3
            output_dim = 1
            data_shape = (1, config["resolution"], config["resolution"], config["resolution"])
            is_point_cloud = True
        elif config["dataset"] == 'era5':
            dataloader = era5(path_to_data=config["path_to_data"],
                              batch_size=config["training"]["batch_size"])
            input_dim = 3
            output_dim = 1
            data_shape = (46, 90)
            is_era5 = True

        # Setup data converter
        if is_point_cloud:
            data_converter = PointCloudDataConverter(device, data_shape, normalize_features=True)
        elif is_era5:
            data_converter = ERA5Converter(device, data_shape, normalize_features=True)
        else:
            data_converter = GridDataConverter(device, data_shape, normalize_features=True)

        checkpoint = torch.load('{}/training_checkpoint_{}.pt'.format(self.path_to_model, self.model_idx), map_location=device)

        epoch = checkpoint['epoch']

        func_dist, optimizer = load_function_distribution_and_optimizer(device,
                                                                             checkpoint['function_distributrion'],
                                                                             config)

        discriminator, optimizer_disc = self._load_discriminator_and_optimizer_disc(device, checkpoint['discriminator'],
                                                                                    config, input_dim, output_dim)

        func_dist.eval()
        func_dist.train()

        discriminator.eval()
        discriminator.train()

        print("\nFunction distribution")
        print(func_dist.hypernetwork)
        print("Number of parameters: {}".format(self.count_parameters(func_dist.hypernetwork)))

        print("\nDiscriminator")
        print(discriminator)
        print("Number of parameters: {}".format(self.count_parameters(discriminator)))

        # Setup trainer
        trainer = CustomTrainer(device=device, function_distribution=func_dist, optimizer=optimizer,
                                discriminator=discriminator, optimizer_disc=optimizer_disc,
                                data_converter=data_converter, epoch=epoch,
                                lr=config["training"]["lr"], lr_disc=config["training"]["lr_disc"],
                                r1_weight=config["training"]["r1_weight"],
                                max_num_points=config["training"]["max_num_points"],
                                print_freq=config["training"]["print_freq"], save_dir=directory,
                                model_save_freq=config["training"]["model_save_freq"],
                                is_voxel=is_voxel, is_point_cloud=is_point_cloud,
                                is_era5=is_era5)
        trainer.train(dataloader, config["training"]["epochs"])
