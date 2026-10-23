# DFR repo audit (before training)

Scope: bottle only. Local source: DFR-master/DFR-master/DFR-source. Original source preserved in source_backup.

| Item | Active official repository setting | Evidence |
|---|---|---|
| Input | RGB 256x256; nearest image resize, ImageNet normalization | MVTec.py NormalDataset/TestDataset |
| Backbone | ImageNet-pretrained VGG19 | vgg19.py VGG19.__init__; vgg.py model_urls |
| Frozen | all backbone requires_grad=False | vgg19.py VGG19.__init__ |
| Layers | relu1_1, relu1_2, relu2_1, relu2_2, relu3_1..relu3_4, relu4_1..relu4_4 | main.py lines 67-69 |
| Scales | 12 selected activations (4 VGG blocks) | main.py active entrypoint overrides parser's 4-layer default |
| Features | [B,3456,64,64] | channel sum 128+256+1024+2048; feature.py Extractor |
| Alignment | bilinear to 256x256, align_corners=True | main.py + feature.py |
| Regional aggregation | 4x4 mean, stride4, dilation1; unfold then mean | feature.py AvgFeatAGG2d |
| CAE | six 1x1 convs: C -> (C+2d)//2 -> 2d -> d -> 2d -> (C+2d)//2 -> C | feat_cae.py FeatCAE |
| BN | enabled after conv 1,2,4,5; ReLU follows | feat_cae.py |
| Loss | mean squared error over all elements | FeatCAE.loss_function |
| Optimizer | Adam; weight_decay=0; PyTorch default betas/eps | anoseg_dfr.py __init__ |
| Learning rate | 1e-4 | main.py line45 |
| Train batch | 4, shuffle=True, drop_last default False | main.py + anoseg_dfr.py |
| Test batch | 1, shuffle=False | anoseg_dfr.py line65 |
| Epoch | executable default 150; same line comment says default700 / wine150 | main.py line46; README has no epoch instructions |
| PCA purpose | determine latent dimension, not transform network inputs | anoseg_dfr.py build_classifier |
| PCA images | first 10 sorted train/good images, eval loader shuffle=False; only first minibatch | anoseg_dfr.py lines66,94-107 |
| PCA features | 10*64*64=40960 observations, 3456 channels | feat_vec + build_classifier |
| PCA variance | PCA(n_components=0.90), actual fitted d to be recorded | anoseg_dfr.py line105 |
| Score | channel mean squared error then bilinear 64x64 -> 256x256, align_corners=True | compute_energy; score |
| Threshold | training-only FPR search; first image per shuffled train minibatch; 100 thresholds; target .005 in config | estimate_thred_with_fpr |
| Reflection | ReflectionPad2d(1) explicitly before VGG convolutions; conv itself no padding | vgg19.py + vgg.py make_layers |
| Metrics | image and pixel AUROC/AP; PRO-AUC; threshold-sweep mean IoU; additional legacy helper metrics | metrics_evaluation; utils.py |
| Pixel ROC-AUC | yes, flattened masks/scores | metrics_evaluation |
| PRO/AUPRO | yes, 5000 thresholds, FPR<=0.3, selected FPR rescaled min-max | metrics_evaluation |
| Image AUROC | yes; image score=max map; label=any positive GT mask | metrics_evaluation |

PAPER SETTING: arXiv v1, section IV-A: 700 epochs, nearest feature alignment; PCA retains90% from random regional sample. Appendix B lists conv/ReLU without BN and intermediate width (C+d)//2. Equations4/5 use L2 norms. TablesII/III bottle 12-scale pixel AUROC=.95, PRO-AUC=.92; 16-scale=.97/.93.

OFFICIAL REPO DEFAULT: 150 epochs, bilinear alignment, BN enabled, intermediate width (C+2d)//2, MSE, PCA first10 images. We preserve this implementation; this is not an exact paper-settings reproduction. Reflection padding is active in repo. Paper boundary analysis distinguishes padding variants; table scores should not be represented as an exact matched reflection-padding baseline.

Paper: https://arxiv.org/html/2012.07122
Author repo: https://github.com/YoungGod/DFR
Online main: https://raw.githubusercontent.com/YoungGod/DFR/master/DFR-source/main.py

No requirements/environment file is present in the DFR source tree. Runner uses a dedicated .venv_dfr; exact installed packages exported to requirements_dfr_current.txt.

Evaluation limitations retained deliberately: image labels derived from GT mask presence; filled_image used for PRO regions; selected FPR min-max normalization instead of an interpolated exact .3 endpoint. Do not claim metric identity with other implementations without checking their definitions.

No training or threshold choice uses test-set performance. The original training validation hook is disabled. Official metric sweep's best IoU is diagnostic only and excluded from the reported selected-threshold performance.
