# BasicLFSR

BasicLFSR is an open-source and easy-to-use **Light Field (LF) image Super-Ressolution (SR)** toolbox based on PyTorch, including a collection of papers on LF image SR and a benchmark to comprehensively evaluate the performance of existing methods.
We also provided simple pipelines to train/valid/test state-of-the-art methods to get started quickly, and you can transform your methods into the benchmark.

**Note**: This repository will be updated on a regular basis, and the pretrained models of existing methods will be open-sourced one after another.
So stay tuned!




## Methods
|  <div style="width: 100pt">   Methods     |   Paper | Repository |
| :-------------: |  :-----: | :-------: |
| LFSSR       | Light Field Spatial Super-Resolution Using Deep Efficient Spatial-Angular Separable Convolution. [TIP2018](https://ieeexplore.ieee.org/abstract/document/8561240) | [spatialsr/<br />DeepLightFieldSSR](https://github.com/spatialsr/DeepLightFieldSSR)|
| resLF       | Residual Networks for Light Field Image Super-Resolution. [CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Residual_Networks_for_Light_Field_Image_Super-Resolution_CVPR_2019_paper.pdf) | [shuozh/resLF](https://github.com/shuozh/resLF)|
| HDDRNet     | High-Dimensional Dense Residual Convolutional Neural Network for Light Field Reconstruction. [TPAMI2019](https://ieeexplore.ieee.org/abstract/document/8854138) | [monaen/<br />LightFieldReconstruction](https://github.com/monaen/LightFieldReconstruction)
| LF-InterNet | Spatial-Angular Interaction for Light Field Image Super-Resolution. [ECCV2019](https://www.researchgate.net/profile/Yingqian-Wang-4/publication/338003771_Spatial-Angular_Interaction_for_Light_Field_Image_Super-Resolution/links/5efeedbd92851c52d61380a2/Spatial-Angular-Interaction-for-Light-Field-Image-Super-Resolution.pdf) | [YingqianWang/<br />LF-InterNet](https://github.com/YingqianWang/LF-InterNet)
| LFSSR-ATO   | Light field spatial super-resolution via deep combinatorial geometry embedding and structural consistency regularization. [CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jin_Light_Field_Spatial_Super-Resolution_via_Deep_Combinatorial_Geometry_Embedding_and_CVPR_2020_paper.pdf) | [jingjin25/<br />LFSSR-ATO](https://github.com/jingjin25/LFSSR-ATO) |
| LF-DFnet    | Light field image super-resolution using deformable convolution. [TIP2020](https://ieeexplore.ieee.org/abstract/document/9286855) | [YingqianWang/<br />LF-DFnet](https://github.com/YingqianWang/LF-DFnet)
| MEG-Net     | End-to-End Light Field Spatial Super-Resolution Network using Multiple Epipolar Geometry. [TIP2021](https://ieeexplore.ieee.org/abstract/document/9465683) | [shuozh/MEG-Net](https://github.com/shuozh/MEG-Net)


## Datasets
We used the EPFL, HCInew, HCIold, INRIA and STFgantry datasets for both training and test. 
Please first download our datasets via [Baidu Drive](https://pan.baidu.com/s/1mYQR6OBXoEKrOk0TjV85Yw) (key:7nzy) or [OneDrive](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/zyliang_stu_xidian_edu_cn/EpkUehGwOlFIuSSdadq9S4MBEeFkNGPD_DlzkBBmZaV_mA?e=FiUeiv), and place the 5 datasets to the folder **`./datasets/`**.

* After downloading, you should find following structure:
  ```
  ./datasets/
      ---> EPFL
          ---> training
              ---> Bench_in_Paris.mat
              ---> Billboards.mat
              ---> ...
          ---> test
              ---> Bikes.mat
              ---> Books__Decoded.mat
              ---> ...
      ---> HCI_new
      ---> ...
  ```


* Run **`Generate_Data_for_Training.py`** to generate training data. The generated data will be saved in **`./data_for_train/`** (SR_5x5_2x, SR_5x5_4x).
* Run **`Generate_Data_for_Test.py`** to generate test data. The generated data will be saved in **`./data_for_test/`** (SR_5x5_2x, SR_5x5_4x).

## Benchmark

We benchmark several methods on above datasets, 
and PSNR and SSIM metrics are used for quantitative evaluation. 


### PSNR and SSIM values achieved by different methods for 2xSR:
|    Method    | Scale |  #Params. | EPFL | HCInew | HCIold | INRIA | STFgantry | Average |
| :----------: | :---: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| Bilinear     | x2 | -- | 28.479949/0.918006 | 30.717944/0.919248 | 36.243278/0.970928 | 30.133901/0.945545 | 29.577468/0.931030 | 31.030508/0.936951 |
| Bicubic      | x2 | -- | 29.739509/0.937581 | 31.887011/0.935637 | 37.685776/0.978536 | 31.331483/0.957731 | 31.062631/0.949769 | 32.341282/0.951851 |
| VDSR         | x2 |
| EDSR         | x2 |    | 33.088922/0.962924 | 34.828374/0.959156 | 41.013989/0.987400 | 34.984982/0.976397 | 36.295865/0.981809 | 
| RCSN         | x2 |    | 
| resLF        | x2 |    | 
| LFSSR        | x2 |    | 33.670594/0.974351 | 36.801555/0.974910 | 43.811050/0.993773 | 35.279443/0.983202 | 37.943969/0.989818 |
| LF-ATO       | x2 |    | 34.271635/0.975711 | 37.243620/0.976684 | 44.205264/0.994202 | 36.169943/0.984241 | 39.636445/0.992862 |
| LF-InterNet  | x2 |    | 
| LF-DFnet     | x2 |    | 
| MEG-Net      | x2 |    | 
| LFT          | x2 |    | 


### PSNR and SSIM values achieved by different methods for 4xSR:

|    Method    | Scale |  #Params. | EPFL | HCInew | HCIold | INRIA | STFgantry | Average |
| :----------: | :---: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| Bilinear     | x4 | -- | 24.567490/0.815793 | 27.084949/0.839677 | 31.688225/0.925630 | 26.226265/0.875682 | 25.203262/0.826105 | 26.954038/0.856577 |
| Bicubic      | x4 | -- | 25.264206/0.832389 | 27.714905/0.851661 | 32.576315/0.934428 | 26.951718/0.886740 | 26.087451/0.845230 | 27.718919/0.870090 |
| VDSR         | x4 |
| EDSR         | x4 |    | 
| RCSN         | x4 |    | 
| resLF        | x4 |    | 
| LFSSR        | x4 |    | 
| LF-ATO       | x4 |    | 
| LF-InterNet  | x4 |    | 
| LF-DFnet     | x4 |    | 
| MEG-Net      | x4 |    | 
| LFT          | x4 |    | 


## Train
* Run **`train.py`** to perform network training. Example for training LFSSR on 5x5 angular resolution for 2x/4x SR:
  ```
  $ python train.py --model_name LFSSR --angRes 5 --scale_factor 2 --batch_size 8
  $ python train.py --model_name LFSSR --angRes 5 --scale_factor 4 --batch_size 4
  ```
* Checkpoints and Logs will be saved to **`./log/`**, and the **`./log/`** has following structure:
  ```
  ./log/
      ---> SR_5x5_2x
          ---> [dataset_name]
              ---> [model_name]
                  ---> [model_name]_log.txt
                  ---> checkpoints
                      ---> [model_name]_5x5_2x_epoch_01_model.pth
                      ---> [model_name]_5x5_2x_epoch_02_model.pth
                      ---> ...
                  ---> results
                      ---> VAL_epoch_01
                      ---> VAL_epoch_02
                      ---> ...
      ---> SR_5x5_4x
  ```

## Test
* Run **`test.py`** to perform network inference. Example for test LFSSR on 5x5 angular resolution for 2x/4xSR:
  ```
  $ python test.py --model_name LFSSR --angRes 5 --scale_factor 2  
  $ python test.py --model_name LFT --angRes 5 --scale_factor 4 
  ```
  
* The PSNR and SSIM values of each dataset will be saved to **`./log/`**, and the **`./log/`** is following structure:
  ```
  ./log/
      ---> SR_5x5_2x
          ---> [dataset_name]
              ---> [model_name]
                  ---> [model_name]_log.txt
                  ---> checkpoints
                      ---> ...
                  ---> results
                      ---> Test
                          ---> evaluation.xls
                          ---> [dataset_1_name]
                              ---> [scene_1_name]
                                  ---> [scene_1_name]_CenterView.bmp
                                  ---> [scene_1_name]_SAI.bmp
                                  ---> views
                                      ---> [scene_1_name]_0_0.bmp
                                      ---> [scene_1_name]_0_1.bmp
                                      ---> ...
                                      ---> [scene_1_name]_4_4.bmp
                              ---> [scene_2_name]
                              ---> ...
                          ---> [dataset_2_name]
                          ---> ...
                      ---> VAL_epoch_01
                      ---> ...
      ---> SR_5x5_4x
  ```


## Recources
We provide some original super-resolved images and useful resources to facilitate researchers to reproduce the above results.



## Other Recources
* [YapengTian/Single-Image-Super-Resolution](https://github.com/YapengTian/Single-Image-Super-Resolution)
* [LoSealL/VideoSuperResolution](https://github.com/LoSealL/VideoSuperResolution)
* [ChaofWang/Awesome-Super-Resolution](https://github.com/ChaofWang/Awesome-Super-Resolution)
* [ptkin/Awesome-Super-Resolution](https://github.com/ptkin/Awesome-Super-Resolution)
* [lightfield-analysis/resources](https://github.com/lightfield-analysis/resources)
* [Joechann0831/LFSRBenchmark](https://github.com/Joechann0831/LFSRBenchmark)
* [YingqianWang/LF-Image-SR](https://github.com/YingqianWang/LF-Image-SR)


## Contact
Any question regarding this work can be addressed to [zyliang@nudt.edu.cn](zyliang@nudt.edu.cn).


