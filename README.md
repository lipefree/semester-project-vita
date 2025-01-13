Install requirements using : `pip install -r CCVPE/requirements.txt`

Install openmmcv for deformable attention : `mim install mmcv`

We have a different models that was implemented in the report. 
To retrain them use : `python CCVPE/{model_name}_train.py`
For example : `python CCVPE/multiple_deformable_attention_train.py`

To run baselines on CCVPE on OSM tile or satellite image, you can use : `python CCVPE/train_VIGOR.py --osm={True,False}` 

maploc files comes from https://github.com/facebookresearch/OrienterNet/tree/main

CCVPE comes from https://github.com/tudelft-iv/CCVPE

Mercator.py and corrected projection from https://github.com/xlwangDev/HC-Net/tree/main/models/utils

References :
- Xia, Z., Booĳ, O., & Kooij, J. (2024). Convolutional Cross-View Pose Estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(5), 3813-3831.
- Paul-Edouard Sarlin, Daniel DeTone, Tsun-Yi Yang, Armen Avetisyan, Julian Straub, Tomasz Malisiewicz, Samuel Rota Bulo, Richard Newcombe, Peter Kontschieder, & Vasileios Balntas (2023). OrienterNet: Visual Localization in 2D Public Maps with Neural Matching. In CVPR.
- Wang, X., Xu, R., Cui, Z., Wan, Z., & Zhang, Y. (2024). Fine-Grained Cross-View Geo-Localization Using a Correlation-Aware Homography Estimator. Advances in Neural Information Processing Systems, 36.
