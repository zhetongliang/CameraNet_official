# CameraNet-Tensorflow
CameraNet: A Two-Stage Framework for Effective Camera ISP Learning

[[Paper](http://www4.comp.polyu.edu.hk/~cslzhang/paper/CameraNet.pdf]]

## Environment
* Tensorflow == 1.15.0
* Cuda == 10.1
* Python == 3.7

## Datasets
* [HDR+](https://drive.google.com/file/d/1IEQc47wyKvo6OniLSXWjHDEhyCYphejX/view?usp=sharing)
* [SID](https://drive.google.com/file/d/1XzgOLtLuKO1giSWnkPshMCdBaVqvtWYu/view?usp=sharing)


## Usage
### For HDR+ ISP
1. Make a dataset directory in the root folder by `mkdir Data`.
2. Download the HDR+ datasets (already including training and testing sets). Unzip it to `Data` folder. Now you should have a folder named `./Data/HDRp`
3. For training, `python train_hdrp.py`
4. For testing, `python test_hdrp.py`
### For SID ISP
1. Make a dataset directory in the root folder by `mkdir Data`.
2. Download the SID datasets (already including training and testing sets). Unzip it to `Data` folder. Now you should have a folder named `./Data/SID`
3. For training, `python train_sid.py`
4. For testing, `python test_sid.py`
Note that for SID, in the paper we use a different training-testing separation of the data from the original SID paper. To allow a beter comparison, in this code we adopt the training-testing separation from the original SID paper. The PSNR and SSIM are a little different from our paper but remain in the same level.

## Contact
Zhetong Liang <zhetong.liang@connect.polyu.hk>

## Citation
@ARTICLE{9329084,  author={Liang, Zhetong and Cai, Jianrui and Cao, Zisheng and Zhang, Lei},  journal={IEEE Transactions on Image Processing},   title={CameraNet: A Two-Stage Framework for Effective Camera ISP Learning},   year={2021},  volume={30},  number={},  pages={2248-2262},  doi={10.1109/TIP.2021.3051486}}
