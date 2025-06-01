This project is managed using [uv](https://docs.astral.sh/uv/):
* clone this repository : `git clone git@github.com:lipefree/semester-project-vita.git && cd semester-project-vita/CCVPE`
* create venv : `uv venv`
* activate venv : `source .venv/bin/activate`
* install mmcv : `mim install mmcv`

To rerun experiments:
* change directory to `semester-project-vita/CCVPE` if not done already
* run script using uv : `uv run run.py`
* To select experiment, modify run.py `experiment_name` field. The list of experiments is located at `registry.py` with short informations.

maploc files comes from https://github.com/facebookresearch/OrienterNet/tree/main

CCVPE comes from https://github.com/tudelft-iv/CCVPE

Mercator.py and corrected projection from https://github.com/xlwangDev/HC-Net/tree/main/models/utils

References :
- Xia, Z., Booĳ, O., & Kooij, J. (2024). Convolutional Cross-View Pose Estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(5), 3813-3831.
- Paul-Edouard Sarlin, Daniel DeTone, Tsun-Yi Yang, Armen Avetisyan, Julian Straub, Tomasz Malisiewicz, Samuel Rota Bulo, Richard Newcombe, Peter Kontschieder, & Vasileios Balntas (2023). OrienterNet: Visual Localization in 2D Public Maps with Neural Matching. In CVPR.
- Wang, X., Xu, R., Cui, Z., Wan, Z., & Zhang, Y. (2024). Fine-Grained Cross-View Geo-Localization Using a Correlation-Aware Homography Estimator. Advances in Neural Information Processing Systems, 36.
