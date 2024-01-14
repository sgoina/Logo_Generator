import os
import torch
import argparse
from PIL import Image
import torchvision.utils as vutils
from texture_utils import Reformer
import time
import numpy as np
import torch.nn.functional as function

import torchvision.transforms as transforms

def transfer(args):

    args.cuda = torch.cuda.is_available()

    if args.compress:
        args.e5 = 'small_models/SE5.pth'
        args.e4 = 'small_models/SE4.pth'
        args.e3 = 'small_models/SE3.pth'
        args.e2 = 'small_models/SE2.pth'
        args.e1 = 'small_models/SE1.pth'

        args.d5 = 'small_models/SD5.pth'
        args.d4 = 'small_models/SD4.pth'
        args.d3 = 'small_models/SD3.pth'
        args.d2 = 'small_models/SD2.pth'
        args.d1 = 'small_models/SD1.pth'

    else:
        args.e5 = 'texture_models/E5.pth'
        args.e4 = 'texture_models/E4.pth'
        args.e3 = 'texture_models/E3.pth'
        args.e2 = 'texture_models/E2.pth'
        args.e1 = 'texture_models/E1.pth'

        args.d5 = 'texture_models/D5.pth'
        args.d4 = 'texture_models/D4.pth'
        args.d3 = 'texture_models/D3.pth'
        args.d2 = 'texture_models/D2.pth'
        args.d1 = 'texture_models/D1.pth'

    # Set up data
    # content/target data
    # if the task is texture transfer, the content/target semantic map must be provided
    if args.style_transfer:
        content = Image.open(args.content).convert('RGB')
        if args.content_size:
            content = transforms.Resize(args.content_size)(content)
        content = transforms.ToTensor()(content).unsqueeze(0)
        if args.content_sem:
            content_sem = Image.open(args.content_sem).convert('RGB')
            if args.content_size:
                content_sem = transforms.Resize(args.content_size)(content_sem)
            content_sem = transforms.ToTensor()(content_sem).unsqueeze(0)
        else:
            content_sem = torch.ones(content.shape)
    else:
        content_sem = Image.open(args.content_sem).convert('RGB')
        if args.content_size:
            content_sem = transforms.Resize(args.content_size)(content_sem)
        content_sem = transforms.ToTensor()(content_sem).unsqueeze(0)
        # for texture transfer, use the content/target semantic map as the content
        content = content_sem


    # style/source data
    style = Image.open(args.style).convert('RGB')
    if args.style_size:
        style = transforms.Resize(args.style_size)(style)
    style = transforms.ToTensor()(style).unsqueeze(0)

    if args.style_sem:
        style_sem = Image.open(args.style_sem).convert('RGB')
        if args.style_size:
            style_sem = transforms.Resize(args.style_size)(style_sem)
        style_sem = transforms.ToTensor()(style_sem).unsqueeze(0)
    else:
        style_sem = torch.ones(style.shape)


    # Set up model and Texture Reformer
    if args.cuda:
        TR = Reformer(args).cuda()
    else:
        TR = Reformer(args)


    # View-Specific Texture Reformation (VSTR) operation
    @torch.no_grad()
    def VSTR(encoder, decoder, content, style, content_sem, style_sem, patch_size, alpha, semantic):
        # make the width and height of the temporary content/target image the same as the content/target semantic map
        if content_sem.shape[2] != content.shape[2] or content_sem.shape[3] != content.shape[3]:
            content = content.squeeze(0).cpu().clone()
            content = transforms.ToPILImage()(content)
            content = transforms.Resize([content_sem.shape[2],content_sem.shape[3]])(content)
            if args.cuda:
                content = transforms.ToTensor()(content).unsqueeze(0).cuda()
            else:
                content = transforms.ToTensor()(content).unsqueeze(0)

        sF  = encoder(style)
        cF  = encoder(content)

        # add the features of the semantic maps
        if semantic == "add":
            sF_sem  = encoder(style_sem)
            cF_sem  = encoder(content_sem)
            csF = TR.VSTR_add(cF, sF, cF_sem, sF_sem, patch_size, alpha, args.add_weight)

        # concatenate the features of the semantic maps
        elif semantic == "concat":
            sF_sem  = encoder(style_sem)
            cF_sem  = encoder(content_sem)
            csF = TR.VSTR_concat(cF, sF, cF_sem, sF_sem, patch_size, alpha, args.concat_weight)

        # directly concatenate with the downsampled semantic maps
        elif semantic == "concat_ds":
            sF_sem  = function.interpolate(style_sem, [sF.size(2), sF.size(3)], mode="nearest")
            cF_sem  = function.interpolate(content_sem, [cF.size(2), cF.size(3)], mode="nearest")
            csF = TR.VSTR_concat(cF, sF, cF_sem, sF_sem, patch_size, alpha, args.concat_weight)
        
        #csF = cF
        Img = decoder(csF)

        return Img



    # Statistic-based Enhancement (SE) operation
    @torch.no_grad()
    def SE(encoder, decoder, content, style, enhance, enhance_alpha):
        sF = encoder(style)
        cF = encoder(content)

        # match the first-order statistics, i.e., channel-wise mean and standard deviation
        if enhance == "adain":
            csF = TR.adain(cF, sF, enhance_alpha)

        # match the second-order statistics, i.e., covariance
        elif enhance == "wct":
            sF  = sF.data.cpu().squeeze(0) # note: svd runs on CPU
            cF  = cF.data.cpu().squeeze(0)
            csF = TR.wct(cF, sF, enhance_alpha)
            if args.cuda:
                csF = csF.cuda()

        #csF = cF
        Img = decoder(csF)
        return Img


    # Run
    if args.cuda:
        content = content.cuda()
        style = style.cuda()
        content_sem = content_sem.cuda()
        style_sem = style_sem.cuda()

    # Start texture/style transfer

    if args.cuda:
        torch.cuda.synchronize() 
    start_time = time.time()

    # Global view structure alignment stage
    print("Processing level 5"); content = VSTR(TR.e5, TR.d5, content, style, content_sem, style_sem, args.coarse_psize, args.coarse_alpha, args.semantic)

    # Local view texture refinement stage
    print("Processing level 4"); content = VSTR(TR.e4, TR.d4, content, style, content_sem, style_sem, args.fine_psize, args.fine_alpha, args.semantic)

    # Holistic effect enhancement stage
    print("Processing level 3"); content = SE(TR.e3, TR.d3, content, style, args.enhance, args.enhance_alpha)
    print("Processing level 2"); content = SE(TR.e2, TR.d2, content, style, args.enhance, args.enhance_alpha)
    print("Processing level 1"); content = SE(TR.e1, TR.d1, content, style, args.enhance, args.enhance_alpha)

    if args.cuda:
        torch.cuda.synchronize()
    end_time = time.time()
    print('Elapsed time is: %.4f seconds' % (end_time - start_time))
    
    out_path = os.path.join(args.outf, args.output)
    vutils.save_image(content.data.cpu(), out_path)

