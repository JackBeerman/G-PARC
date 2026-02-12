import torch
import torch.nn as nn

# Import the custom graph components
# Ensure these import paths match your file structure
from utilities.spade import GraphSPADEGeneratorUnit
from utilities.resnet import GraphResNet


class MappingAndRecon(nn.Module):
    """
    Graph-based Mapping and Reconstruction module.
    
    Updated to support Zero-Initialization for stability.
    """
    def __init__(
        self,
        n_base_features=128,
        n_mask_channel=1,
        output_channel=1,
        heads=4,
        concat=True,
        dropout=0.0,
        add_noise=True,
        zero_init=True,  # <--- NEW ARGUMENT (Default True for stability)
    ):
        super().__init__()
        self.add_noise = add_noise

        # Initialize SPADE generator unit
        self.spade = GraphSPADEGeneratorUnit(
            in_channels=n_base_features,
            out_channels=n_base_features,
            mask_channels=n_mask_channel,
            heads=heads,
            concat=concat,
            dropout=dropout,
            add_noise=add_noise, # Pass noise config to module
            zero_init=zero_init  # Pass zero_init config to module
        )

        # Initialize ResNet block
        self.resnet = GraphResNet(
            in_channels=n_base_features,
            block_dimensions=[n_base_features, n_base_features],
            heads=heads,
            concat=concat,
            dropout=dropout
        )

        # Final convolution layer (Linear = graph equivalent of Conv2d with kernel=1)
        self.conv_out = nn.Linear(n_base_features, output_channel)

    def forward(self, dynamic_feature, advec_diff, edge_index):
        """
        Forward pass.
        """
        # Note: add_noise logic is now handled internally by the SPADE unit
        # based on how we initialized it, but we pass edge_index for the GATs.
        spade_out = self.spade(dynamic_feature, advec_diff, edge_index)
        
        resnet_out = self.resnet(spade_out, edge_index)
        conv_out = self.conv_out(resnet_out)
        
        return conv_out