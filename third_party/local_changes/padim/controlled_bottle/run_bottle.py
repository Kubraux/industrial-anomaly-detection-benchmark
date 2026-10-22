import random
import time, traceback, shutil
from random import sample
import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18, Wide_ResNet50_2_Weights, ResNet18_Weights
import mvtec_controlled as mvtec
from pathlib import Path
from independent_metrics import audit_data, save_evaluation, dump, sha, choose_visual_examples


# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument('--data_path', type=str, default=str(Path(__file__).resolve().parents[2] / 'archive'))
    parser.add_argument('--save_path', type=str, default=str(Path(__file__).resolve().parents[2] / 'PaDiM_bottle_WR50_Rd550'))
    parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2'], default='wide_resnet50_2')
    parser.add_argument('--execute', action='store_true', help='Run only after explicit experiment approval')
    return parser.parse_args()


def main():

    args = parse_args()
    if not args.execute:
        print('Prepared only. No model loading, feature extraction or Gaussian fitting. Explicit approval is required before --execute.')
        return
    run_path = Path(args.save_path).resolve()
    run_path.mkdir(parents=True, exist_ok=False)
    manifest = audit_data(Path(args.data_path), 'bottle')
    dump(run_path / 'manifest.json', manifest)
    dump(run_path/'visual_selection.json', choose_visual_examples(manifest, seed=1024))
    protocol = __import__('json').loads((Path(__file__).parent/'protocol.json').read_text())
    protocol.update(status='running', backbone=args.arch, selected_dimensions=550 if args.arch=='wide_resnet50_2' else 100)
    dump(run_path/'protocol.json', protocol)
    dump(run_path/'source_hashes.json', {p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    dump(run_path/'status.json', {'stage':'running','category':'bottle'})
    snapshot=run_path/'source_snapshot'
    snapshot.mkdir()
    for src in Path(__file__).parent.iterdir():
        if src.suffix in ('.py','.json','.md'): shutil.copy2(src,snapshot/src.name)
    started=time.perf_counter()
    torch.set_num_threads(4)
    np.random.seed(1024)

    # load model
    if args.arch == 'resnet18':
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1, progress=True)
        t_d = 448
        d = 100
    elif args.arch == 'wide_resnet50_2':
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1, progress=True)
        t_d = 1792
        d = 550
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    random.seed(1024)
    torch.manual_seed(1024)
    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    idx = torch.tensor(sample(range(0, t_d), d))
    dump(run_path/'selected_channels.json', idx.tolist())

    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    os.makedirs(os.path.join(args.save_path, 'temp_%s' % args.arch), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    total_roc_auc = []
    total_pixel_roc_auc = []

    for class_name in ['bottle']:

        train_dataset = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
        test_dataset = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)

        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        # extract train set features
        train_feature_filepath = os.path.join(args.save_path, 'temp_%s' % args.arch, 'train_%s.pkl' % class_name)
        if not os.path.exists(train_feature_filepath):
            for (x, _, _) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
                # model prediction
                with torch.no_grad():
                    _ = model(x.to(device))
                assert len(outputs)==3
                assert all(torch.isfinite(v).all() for v in outputs)
                assert not model.training and not any(p.requires_grad for p in model.parameters())
                dump(run_path/'status.json', {'stage':'train_feature_extraction','category':'bottle',
                     'batches_completed':len(train_outputs['layer1'])+1,'total_batches':len(train_dataloader)})
                # get intermediate layer outputs
                for k, v in zip(train_outputs.keys(), outputs):
                    train_outputs[k].append(v.cpu().detach())
                # initialize hook outputs
                outputs = []
            for k, v in train_outputs.items():
                train_outputs[k] = torch.cat(v, 0)

            # Embedding concat
            embedding_vectors = train_outputs['layer1']
            for layer_name in ['layer2', 'layer3']:
                embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])

            # randomly select d dimension
            embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
            # calculate multivariate Gaussian distribution
            B, C, H, W = embedding_vectors.size()
            embedding_vectors = embedding_vectors.view(B, C, H * W)
            mean = torch.mean(embedding_vectors, dim=0).numpy()
            cov = torch.zeros(C, C, H * W).numpy()
            I = np.identity(C)
            probe_cov=np.cov(embedding_vectors[:,:,0].numpy(),rowvar=False)+0.01*I
            np.linalg.cholesky(probe_cov)
            assert np.isfinite(probe_cov).all() and np.isfinite(mean).all()
            dump(run_path/'smoke_checks.json',dict(passed=True,feature_shape=[B,C,H,W],
                 frozen_backbone=True,finite_features=True,regularized_covariance_positive_definite=True,
                 train_only=True))
            print('Train-only numerical checks passed',flush=True)
            for i in range(H * W):
                if i%64==0:
                    dump(run_path/'status.json',dict(stage='gaussian_fitting',position=i,total=H*W))
                # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_
                cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
            # save learned distribution
            train_outputs = [mean, cov]
            with open(train_feature_filepath, 'wb') as f:
                pickle.dump(train_outputs, f)
        else:
            print('load train set feature from: %s' % train_feature_filepath)
            with open(train_feature_filepath, 'rb') as f:
                train_outputs = pickle.load(f)

        fit_seconds=time.perf_counter()-started
        dump(run_path/'timing.json',dict(fitting_including_setup_seconds=fit_seconds))
        dump(run_path/'status.json',dict(stage='test_feature_extraction'))
        inference_started=time.perf_counter()
        gt_list = []
        gt_mask_list = []
        test_imgs = []

        # extract test set features
        for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())
            # model prediction
            with torch.no_grad():
                _ = model(x.to(device))
            # get intermediate layer outputs
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            outputs = []
        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)
        
        # Embedding concat
        embedding_vectors = test_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])

        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
        
        # calculate distance matrix
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
        dist_list = []
        for i in range(H * W):
            if i%64==0:
                dump(run_path/'status.json',dict(stage='mahalanobis_scoring',position=i,total=H*W))
            mean = train_outputs[0][:, i]
            conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)

        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample
        dist_list = torch.tensor(dist_list)
        score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bicubic',
                                  align_corners=False).squeeze(1).numpy()
        
        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)
        
        # Raw scores: no test-dependent scaling, no test-selected decision threshold.
        scores = score_map.copy()
        assert np.isfinite(scores).all()

        inference_seconds=time.perf_counter()-inference_started
        dump(run_path/'timing.json',dict(fitting_including_setup_seconds=fit_seconds,
            inference_all_images_seconds=inference_seconds,
            inference_seconds_per_image=inference_seconds/len(gt_list),
            scope='test loading, features, Mahalanobis, resize and smoothing; excludes metrics'))
        dump(run_path/'status.json',dict(stage='metrics'))
        # calculate image-level ROC AUC score
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        fpr, tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        total_roc_auc.append(img_roc_auc)
        print('image ROCAUC: %.3f' % (img_roc_auc))
        fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc))
        
        gt_mask = np.asarray(gt_mask_list) > 0

        # calculate per-pixel level ROCAUC
        fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        total_pixel_roc_auc.append(per_pixel_rocauc)
        print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

        fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
        save_dir = args.save_path + '/' + f'pictures_{args.arch}'
        os.makedirs(save_dir, exist_ok=True)
        save_evaluation(run_path, scores, gt_list, gt_mask, test_dataset.x)


    print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.legend(loc="lower right")

    print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)
    dump(run_path/'status.json',dict(stage='complete',category='bottle'))


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    
    return x


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


if __name__ == '__main__':
    try:
        main()
    except Exception:
        args=parse_args()
        path=Path(args.save_path)
        if args.execute and path.exists():
            dump(path/'status.json',dict(stage='failed',error=traceback.format_exc()))
        raise
