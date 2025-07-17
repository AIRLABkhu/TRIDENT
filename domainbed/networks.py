# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


from diffusers.models.autoencoders import AutoencoderTiny
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

from domainbed.lib import wide_resnet


def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x

class DinoV2(torch.nn.Module):
    """ """
    def __init__(self,input_shape, hparams):
        super(DinoV2, self).__init__()

        self.network = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.n_outputs =  5 * 768

        nc = input_shape[0]

        if nc != 3:
            raise RuntimeError("Inputs must have 3 channels")

        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['vit_dropout'])

        if hparams["vit_attn_tune"]:
            for n,p in self.network.named_parameters():
                if 'attn' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False


    def forward(self, x):
        x = self.network.get_intermediate_layers(x, n=4, return_class_token=True)
        linear_input = torch.cat([
            x[0][1],
            x[1][1],
            x[2][1],
            x[3][1],
            x[3][0].mean(1)
            ], dim=1)
        return self.dropout(linear_input)
    
class ViT_base(torch.nn.Module):
    """ """
    def __init__(self,input_shape, hparams):
        super(ViT_base, self).__init__()
        self.network = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.n_outputs =  768

        nc = input_shape[0]

        if nc != 3:
            raise RuntimeError("Inputs must have 3 channels")

        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['vit_dropout'])

        if hparams["vit_attn_tune"]:
            for n,p in self.network.named_parameters():
                if 'attn' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        del self.network.head
        self.network.head = Identity()

    def forward(self, x):
        out = self.network.forward(x)
        return out
    
class ViT_small(torch.nn.Module):
    """ """
    def __init__(self,input_shape, hparams):
        super(ViT_small, self).__init__()
        self.network = timm.create_model('vit_small_patch16_224', pretrained=True)
        self.n_outputs =  384

        nc = input_shape[0]

        if nc != 3:
            raise RuntimeError("Inputs must have 3 channels")

        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['vit_dropout'])

        if hparams["vit_attn_tune"]:
            for n,p in self.network.named_parameters():
                if 'attn' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        del self.network.head
        self.network.head = Identity()

    def forward(self, x):
        out = self.network.forward(x)
        return out

class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams['resnet18']:
            self.network = torchvision.models.resnet18(pretrained=True)
            # self.network = timm.create_model('resnet18.tv_in1k', pretrained=True)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(pretrained=True)
            # self.network = timm.create_model('resnet50.tv_in1k', pretrained=True)
            self.n_outputs = 2048

        if hparams['resnet50_augmix']:
            self.network = timm.create_model('resnet50.ram_in1k', pretrained=True)
            self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        if hparams["freeze_bn"]:
            self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.hparams["freeze_bn"]:
            self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    elif input_shape[1:3] == (224, 224):
        if hparams["vit"]:
            if hparams["dinov2"]:
                return DinoV2(input_shape, hparams)
            if hparams["small"]:
                return ViT_small(input_shape, hparams)
            else:
                return ViT_base(input_shape, hparams)
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


class timm_Renset(torch.nn.Module):
    def __init__(self, input_shape, hparams, num_classes, split_layer):
        super(timm_Renset, self).__init__()
        if hparams['resnet18']:
            self.network = timm.create_model('resnet18.tv_in1k', pretrained=True, drop_rate=hparams['resnet_dropout'])
        else:
            self.network = timm.create_model('resnet50.tv_in1k', pretrained=True, drop_rate=hparams['resnet_dropout'])
        if hparams['resnet50_augmix']:
            self.network = timm.create_model('resnet50.ram_in1k', pretrained=True, drop_rate=hparams['resnet_dropout'])
        self.network.reset_classifier(num_classes)

        self.hparams = hparams
        self.split_layer = split_layer
        self.inter_shape = [112, 56, 28, 14, 7]
        self.inter_feat_dim = self.network.feature_info[split_layer]['num_chs']

        if hparams["load_pretrained"] is not None:
            self.load_pretrained(hparams["load_pretrained"])

        self._init_layer()
        self._init_param_grad()

    def _init_layer(self):
        layers = [self.network.layer1, self.network.layer2, self.network.layer3, self.network.layer4]
        self.featurizer = nn.Sequential(
            self.network.conv1,
            self.network.bn1, 
            self.network.act1,
            self.network.maxpool,
            *layers[:self.split_layer]
        )
        self.classifier = nn.Sequential(
            *layers[self.split_layer:],
            self.network.global_pool,
            self.network.fc,
        )

    def _init_param_grad(self):
        if self.hparams['learnable_param'] == 'whole':
            for name, param in self.featurizer.named_parameters():
                param.requires_grad = True
            for name, param in self.classifier.named_parameters():
                param.requires_grad = True
        elif self.hparams['learnable_param'] == 'classifier': 
            for name, param in self.featurizer.named_parameters():
                param.requires_grad = False
            for name, param in self.classifier.named_parameters():
                param.requires_grad = True
        elif self.hparams['learnable_param'] == 'header':
            for name, param in self.featurizer.named_parameters():
                param.requires_grad = False
            classifier_params = list(self.classifier.named_parameters())
            for name, param in classifier_params[:-2]:
                param.requires_grad = False
            for name, param in classifier_params[-2:]:
                param.requires_grad = True
        elif self.hparams['learnable_param'] == 'onlydiff':
            for name, param in self.featurizer.named_parameters():
                param.requires_grad = False
            for name, param in self.classifier.named_parameters():
                param.requires_grad = False

    def parameters(self):
        return (list(self.featurizer.parameters()) + list(self.classifier.parameters()))

    def forward(self, x):
        return self.classifier(self.featurizer(x))

    def train(self, mode=True):
        super().train(mode)
        if self.hparams["freeze_bn"]:
            self.freeze_bn()
    
    def freeze_bn(self):
        for m in self.featurizer.modeuls():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def load_pretrained(self, path):
        pre_ckpt = torch.load(path, map_location='cpu')
        self.network.load_state_dict(pre_ckpt)
        print("Sucess to load pre_trained model")

def timm_network(input_shape, hparams, num_classes, split_layer):
    if input_shape[1:3] == (224, 224):
        return timm_Renset(input_shape, hparams, num_classes, split_layer)
    else:
        raise NotImplementedError

class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)

# for ddpm

def ddpm_schedules(beta1, beta2, T, device=None):
    '''
    Returns pre-computed schedules for DDPM sampling, training process.
    '''
    assert beta1 < beta2 <= 1.0, 'beta1 and beta2 must be in (0, 1)'
    if device is None:
        device = 'cpu'

    beta_t = torch.linspace(beta1, beta2, T)

    alpha_t = 1 - beta_t
    sqrt_beta_t = torch.sqrt(beta_t)
    alphabar_t = torch.cumprod(alpha_t, dim=0)

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1.0 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        'beta_t': beta_t.unsqueeze(-1).to(device),  # \beta_t
        'alpha_t': alpha_t.unsqueeze(-1).to(device),  # \alpha_t
        'oneover_sqrta': oneover_sqrta.unsqueeze(-1).to(device),  # 1/\sqrt{\alpha_t}
        'sqrt_beta_t': sqrt_beta_t.unsqueeze(-1).to(device),  # \sqrt{\beta_t}
        'alphabar_t': alphabar_t.unsqueeze(-1).to(device),  # \bar{\alpha_t}
        'sqrtab': sqrtab.unsqueeze(-1).to(device),  # \sqrt{\bar{\alpha_t}}
        'sqrtmab': sqrtmab.unsqueeze(-1).to(device),  # \sqrt{1-\bar{\alpha_t}}
        'mab_over_sqrtmab': mab_over_sqrtmab_inv.unsqueeze(-1).to(device),  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }

class LatentDiffusion(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        input_channels: int, 
        steps: int=500, 
        betas: tuple[float, float]=(1.0E-4, 2.0E-2), 
        ae_enable = False,
    ): 
        super(LatentDiffusion, self).__init__()
        # assert input_size >= 64 and float.is_integer(math.log2(latent_channels))
        ae_kwargs = dict()
        unet_kwargs = dict()
        self.ae_enable = ae_enable

        # for AE kwargs
        if self.ae_enable:
            ae_kwargs['latent_channels'] = input_channels // 8
            ae_kwargs['in_channels'] = input_channels
            ae_kwargs['out_channels'] = input_channels
            ae_kwargs['encoder_block_out_channels'] = (input_channels, input_channels//2, input_channels//4)
            ae_kwargs['decoder_block_out_channels'] = (input_channels//4, input_channels//2, input_channels)
            ae_kwargs['num_encoder_blocks'] = (1, 1, 1)
            ae_kwargs['num_decoder_blocks'] = (1, 1, 1)
            latent_channels = input_channels//8
        else:
            latent_channels = input_channels

        unet_kwargs['sample_size'] = (input_size, input_size)
        unet_kwargs['in_channels'] = latent_channels
        unet_kwargs['out_channels'] = latent_channels
        unet_kwargs['down_block_types'] = ("CrossAttnDownBlock2D",  "DownBlock2D")
        unet_kwargs['up_block_types'] = ("UpBlock2D", "CrossAttnUpBlock2D")
        unet_kwargs['block_out_channels'] = (latent_channels, latent_channels*2)
        unet_kwargs['layers_per_block'] = 2
        unet_kwargs['cross_attention_dim'] = latent_channels
        unet_kwargs['attention_head_dim'] = latent_channels // 8
        unet_kwargs['encoder_hid_dim'] = latent_channels
        unet_kwargs['use_linear_projection'] = True
        unet_kwargs['addition_embed_type'] = "image"
        # unet_kwargs['addition_time_embed_dim'] = 128
        unet_kwargs['time_embedding_dim'] = latent_channels
        # unet_kwargs['time_cond_proj_dim'] = latent_channels


        self.autoencoder = AutoencoderTiny(**ae_kwargs) 
        self.unet = UNet2DConditionModel(**unet_kwargs)

        self.steps = steps
        self.betas = betas
        self.schedule_values = nn.ParameterDict(ddpm_schedules(*betas, steps))

    def parameters(self):
        if self.ae_enable:
            return (list(self.autoencoder.parameters()) + list(self.unet.parameters()))
        else:
            return list(self.unet.parameters())

    def forward_process(self, x: torch.Tensor, z: torch.Tensor, t: int|torch.Tensor):
        if isinstance(t, int):
            t = torch.tensor([t], dtype=torch.long, device=x.device)
        assert t.ndim == 1
        if t.size(0) != x.size(0):
            if t.size(0) == 1:
                t = t.expand((x.size(0),))
            else:
                raise ValueError()
        sqrtab = self.schedule_values['sqrtab'][t][:,None, None].to(x.device)
        sqrtmab = self.schedule_values['sqrtmab'][t][:,None, None].to(x.device)
        return (sqrtab * x) + (sqrtmab * z)

        # return x * alphabar_t + z * betabar_t
    
    def reverse_process(self, x_t, t, c):
        z_hat = self.forward_unet(x_t, t, c) # predicted noise

        oneover_sqrta = self.schedule_values['oneover_sqrta'][t].to(z_hat.device)
        mab_over_sqrtmab = self.schedule_values['mab_over_sqrtmab'][t].to(z_hat.device)
        sqrt_beta_t = self.schedule_values['sqrt_beta_t'][t].to(z_hat.device)

        x_tm1 = oneover_sqrta * (x_t - z_hat * mab_over_sqrtmab) 
        x_tm1 = x_tm1 + sqrt_beta_t * torch.randn_like(x_tm1)
        return x_tm1
    
    def loss_fn(self, f_0, f_T):
        batch_size = f_0.size(0)
        r_0 = f_T - f_0
        noise = torch.randn(r_0.shape, device=f_0.device)
        t = torch.randint(0, self.steps, size=(batch_size,), device=f_0.device)
        r_t = self.forward_process(r_0, noise, t) # add noise

        pred_r_0 = self.forward_unet(r_t, t, f_T)
        loss = F.mse_loss(pred_r_0, r_0)
        return loss
        
    def forward_encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x).latents
    
    # pred noise
    def forward_unet(self, x: torch.Tensor, t: int|float|torch.Tensor, c: torch.Tensor|None=None):
        c = c.view(c.size(0), c.size(1), -1)
        x = x.view(x.size(0), x.size(1), -1)
        return self.unet.forward(x, timestep=t, encoder_hidden_states=c, added_cond_kwargs={"image_embeds": c.mean(1)}).sample
    
    def forward_decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(x).sample
    
    # Pseudo Code
    def forward_train(self, x_0, x_T):
        """
        x_0 : non-augmented feature
        x_T : augmented feature
        we pupose to get denoising vector that make x_T to x_0.
        """
        # encode feature to latent space
        batch_size = x_0.size(0)

        #ae
        f_0 = self.forward_encode(x_0) if self.ae_enable else x_0 # origin
        f_T = self.forward_encode(x_T) if self.ae_enable else x_T # augmented

        rec_x_0 = self.forward_decode(f_0) if self.ae_enable else None

        ddpm_loss = self.loss_fn(f_0.detach(), f_T.detach())
        rec_loss = F.mse_loss(rec_x_0, x_0) if self.ae_enable else 0
        
        return ddpm_loss, rec_loss 

    def forward_pred(self, x):
        # only one feat come in
        f_x = self.forward_encode(x) if self.ae_enable else x
        z = torch.randn(f_x.shape)
        for step in torch.arange(self.steps-1, -1, -1):
            z = self.reverse_process(z.to(f_x.device), step, f_x)
        refined = f_x - z
        refined_x = self.forward_decode(refined) if self.ae_enable else refined
        return refined_x


class ResLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, residual: bool=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = residual

        self.linear1 = nn.Linear(in_dim, out_dim)
        self.activation1 = nn.GELU()
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.activation2 = nn.GELU()

        if in_dim != out_dim:
            self.residual_layer = nn.Linear(in_dim, out_dim)
        else:
            self.residual_layer = nn.Identity()

    def forward(self, x):
        identity = x

        out = self.linear1(x)
        out = self.activation1(out)
        out = self.linear2(out)
        out = self.activation2(out)

        if self.residual:
            return out + self.residual_layer(identity)
        else:
            return out
            
class Unet(nn.Module):
    def __init__(self, dims=[256, 256, 256], bottleneck=False):
        super().__init__()
        
        self.dims = dims
        self.bottleneck = bottleneck
        
        # Encoder
        self.encoder_layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.encoder_layers.append(nn.Sequential(
                ResLinear(dims[i], dims[i]),
                nn.Linear(dims[i], dims[i+1]),
                nn.GELU()
            ))
        
        # Bottleneck
        if bottleneck:
            self.bottleneck_layer = ResLinear(dims[-1], dims[-1])
        else:
            self.bottleneck_layer = nn.Identity()
        
        # Decoder
        self.decoder_layers = nn.ModuleList()
        for i in range(len(dims) - 1, 0, -1):
            self.decoder_layers.append(nn.Sequential(
                ResLinear(dims[i] * 2, dims[i]),  # *2 for skip connection
                nn.Linear(dims[i], dims[i-1]),
                nn.GELU()
            ))
        
        # Final layer
        self.final_layer = ResLinear(dims[0], dims[0])
        
    def forward(self, x):
        # Encoding
        skip_connections = []
        for layer in self.encoder_layers:
            x = layer(x)
            skip_connections.append(x)
        
        # Bottleneck
        x = self.bottleneck_layer(x)
        
        # Decoding
        for i, layer in enumerate(self.decoder_layers):
            skip = skip_connections[-(i+1)]
            x = torch.cat([x, skip], dim=-1)  # Skip connection
            x = layer(x)
        
        # Final layer
        x = self.final_layer(x)
        
        return x

class SimpleLatentDiffusion(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        input_channels: int, 
        steps: int=500, 
        betas: tuple[float, float]=(1.0E-4, 2.0E-2), 
        ae_enable = True,
    ): 
        super().__init__()
        if ae_enable:
            latent_channels = input_channels // 2
            self.encoder = ResLinear(input_channels, latent_channels)
            self.decoder = ResLinear(latent_channels, input_channels)
        else:
            latent_channels = input_channels
            self.encoder = nn.Identity()
            self.decoder = nn.Identity()

        self.unet = Unet([latent_channels, latent_channels * 2, latent_channels], bottleneck=True)
        self.t_emb = nn.Embedding(steps, input_size**2)
        self.c_emb_layer = nn.Sequential(nn.Linear(latent_channels, latent_channels), nn.GELU())
        self.init_layer = nn.Sequential(nn.Linear(latent_channels + 1, latent_channels), nn.GELU())
        
        self.steps = steps
        self.betas = betas
        self.ae_enable = ae_enable
        self.schedule_values = nn.ParameterDict(ddpm_schedules(*betas, steps))
    
    def to(self, device):
        self.encoder.to(device)
        self.decoder.to(device)
        self.unet.to(device)
        self.t_emb.to(device)
        self.c_emb_layer.to(device)
        self.init_layer.to(device)
        # self.schedule_values.to(device)

    def parameters(self):
        parameters = list(self.unet.parameters()) + \
             list(self.t_emb.parameters()) + \
             list(self.c_emb_layer.parameters()) + \
             list(self.init_layer.parameters())
        if self.ae_enable:
            return (list(self.encoder.parameters()) + list(self.decoder.parameters()) + parameters)
        else:
            return parameters

    def forward_process(self, x: torch.Tensor, z: torch.Tensor, t: int|torch.Tensor):
        if isinstance(t, int):
            t = torch.tensor([t], dtype=torch.long, device=x.device)
        assert t.ndim == 1
        if t.size(0) != x.size(0):
            if t.size(0) == 1:
                t = t.expand((x.size(0),))
            else:
                raise ValueError()
        # breakpoint()
        sqrtab = self.schedule_values['sqrtab'][t][:,None].to(x.device)
        sqrtmab = self.schedule_values['sqrtmab'][t][:,None].to(x.device)
        return (sqrtab * x) + (sqrtmab * z)

        # return x * alphabar_t + z * betabar_t
    
    def reverse_process(self, x_t, t, c):
        # if isinstance(t, int):
        #     t = torch.tensor([t], dtype=torch.long, device=x.device)
        t = t.unsqueeze(-1).expand((x_t.size(0),))
        z_hat = self.forward_unet(x_t, t, c) # predicted noise

        oneover_sqrta = self.schedule_values['oneover_sqrta'][t][:,None].to(z_hat.device)
        mab_over_sqrtmab = self.schedule_values['mab_over_sqrtmab'][t][:,None].to(z_hat.device)
        sqrt_beta_t = self.schedule_values['sqrt_beta_t'][t][:,None].to(z_hat.device)
        # breakpoint()
        x_tm1 = oneover_sqrta * (x_t - z_hat * mab_over_sqrtmab) 
        x_tm1 = x_tm1 + sqrt_beta_t * torch.randn_like(x_tm1)
        return x_tm1
    
    def loss_fn(self, f_0, f_T):
        batch_size = f_0.size(0)
        r_0 = f_T - f_0
        noise = torch.randn(r_0.shape, device=f_0.device)
        # t = torch.randint(0, self.steps, size=(batch_size,), device=f_0.device)
        t = torch.randint(0, self.steps, size=(batch_size,))
        r_t = self.forward_process(r_0, noise, t) # add noise

        pred_r_0 = self.forward_unet(r_t, t.to(r_t.device), f_T)
        loss = F.mse_loss(pred_r_0, r_0)
        return loss
        
    def forward_encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)
        return self.encoder(x)
    
    # pred noise
    def forward_unet(self, x: torch.Tensor, t: int|float|torch.Tensor, c: torch.Tensor|None=None):
        # c = c.view(c.size(0), c.size(1), -1).permute(0, 2, 1)
        c_emb = self.c_emb_layer(c)
        t_emb = self.t_emb(t.to(c.device))
        t_emb = t_emb.unsqueeze(-1)
        x_emb = torch.cat([x, t_emb], dim=-1)
        x_emb = self.init_layer(x_emb) + c_emb
        x_emb = self.unet(x_emb)
        return x_emb
    
    def forward_decode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)
        B, D, C = x.shape
        d = int(math.sqrt(D))
        x = x.permute(0, 2, 1)
        x = x.view(B, C, d, d)
        return x
    
    # Pseudo Code
    def forward_train(self, x_0, x_T):
        """
        x_0 : non-augmented feature
        x_T : augmented feature
        we pupose to get denoising vector that make x_T to x_0.
        """
        # encode feature to latent space
        batch_size = x_0.size(0)

        #ae
        f_0 = self.forward_encode(x_0)# origin
        f_T = self.forward_encode(x_T) # augmented

        rec_x_0 = self.forward_decode(f_0) if self.ae_enable else None

        ddpm_loss = self.loss_fn(f_0.detach(), f_T.detach())
        rec_loss = F.mse_loss(rec_x_0, x_0) if self.ae_enable else 0
        
        return ddpm_loss, rec_loss 

    def forward_pred(self, x):
        # only one feat come in
        f_x = self.forward_encode(x) if self.ae_enable else x
        z = torch.randn(f_x.shape)
        for step in torch.arange(self.steps-1, -1, -1):
            z = self.reverse_process(z.to(f_x.device), step, f_x)
        refined = f_x - z
        refined_x = self.forward_decode(refined) if self.ae_enable else refined
        return refined_x
