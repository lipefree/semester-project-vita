from model_utils.multimae import MultiMAE, pretrain_multimae_base
from model_utils.input_adapters import PatchedInputAdapter
from model_utils.output_adapters import SpatialOutputAdapter
import torch
import torch.nn as nn
import os

class MultiMaeEncoder(nn.Module):
    def __init__(self, patch_size=32, use_pretrain=False):
        super().__init__()

        self.patch_size = patch_size

        self.input_adapters = {
            'osm': PatchedInputAdapter(
                num_channels=3,
                stride_level=1,
                patch_size_full=patch_size,
                image_size=512
            ),
            'sat': PatchedInputAdapter(
                num_channels=3,
                stride_level=1,
                patch_size_full=patch_size,
                image_size=512
            ),
        }

        context_tasks = ['osm', 'sat']
        self.output_adapters = { 
            'osm': SpatialOutputAdapter(
                num_channels=3,
                stride_level=1,
                patch_size_full=patch_size,
                dim_tokens_enc=768,
                context_tasks=context_tasks
            ),
            'sat': SpatialOutputAdapter(
                num_channels=3,
                stride_level=1,
                patch_size_full=patch_size,
                dim_tokens_enc=768,
                context_tasks=context_tasks
            )
        }

        self.multimae = pretrain_multimae_base(input_adapters=self.input_adapters,
                                 output_adapters=self.output_adapters)

        self.multimae.load_state_dict(
                                      torch.load('./weights/multimae-b_98_rgb+-depth-semseg_1600e_multivit-afff3f8c.pth'),
                                      strict=False)

    def forward(self, grd, sat, osm, task_masks=None, alphas=float('inf'), num_encoded_tokens=255, masked_input=True):
        x = {'osm':osm, 'sat':sat}
        preds, masks, encoder_tokens = self.multimae(x,
                               num_encoded_tokens=num_encoded_tokens,
                               mask_inputs=True,
                               task_masks=task_masks,
                               alphas=alphas
                           )

        return preds, masks, encoder_tokens

    def save_multimae_weights(self, label, epoch, ):
        scratch_path = '/work/vita/qngo'
        model_name = 'models/VIGOR/'+label+'/' + str(epoch) + '/'
        model_dir = os.path.join(scratch_path, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(self.multimae.cpu().state_dict(), model_dir+'model.pt') # saving model
        print('model saved')

    def load_multimaae_weights(self, weights_dir):
        self.multimae.load_state_dict(
                              torch.load(weights_dir),
                              strict=True)
        print('pre-trained weights loaded')
