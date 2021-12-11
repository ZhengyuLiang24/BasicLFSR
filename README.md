# BasicLFSR

### <img src="https://raw.github.com/ZhengyuLiang24/BasicLFSR/main/figs/Thumbnail.jpg" width="1000">

**BasicLFSR is a PyTorch-based open-source and easy-to-use toolbox for Light Field (LF) image Super-Ressolution (SR). This toolbox 
introduces a simple pipeline to train/test your methods, and builds a benchmark to comprehensively evaluate the performance of existing methods.
Our BasicLFSR can help researchers to get access to LF image SR quickly, and facilitates the development of novel methods. Welcome to contribute your own methods to the benchmark.**

**Note: This repository will be updated on a regular basis. Please stay tuned!**

## Contributions
* **We provide a PyTorch-based open-source and easy-to-use toolbox for LF image SR.**
* **We re-implement a number of existing methods on the unified datasets, and develop a benchmark for performance evaluation.**
* **We share the codes, models and results of existing methods to help researchers better get access to this area.**


## News & Updates
* **Dec 10, 2021: Add the comparisions of model sizes of existing methods.**


## Datasets
**We used the EPFL, HCInew, HCIold, INRIA and STFgantry datasets for both training and test. 
Please first download our datasets via [Baidu Drive](https://pan.baidu.com/s/1mYQR6OBXoEKrOk0TjV85Yw) (key:7nzy) or [OneDrive](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/zyliang_stu_xidian_edu_cn/EpkUehGwOlFIuSSdadq9S4MBEeFkNGPD_DlzkBBmZaV_mA?e=FiUeiv), and place the 5 datasets to the folder `./datasets/`.**

* **Our project has the following structure:**
  ```
  ├──./datasets/
  │    ├── EPFL
  │    │    ├── training
  │    │    │    ├── Bench_in_Paris.mat
  │    │    │    ├── Billboards.mat
  │    │    │    ├── ...
  │    │    ├── test
  │    │    │    ├── Bikes.mat
  │    │    │    ├── Books__Decoded.mat
  │    │    │    ├── ...
  │    ├── HCI_new
  │    ├── ...
  ```
* **Run `Generate_Data_for_Training.m` to generate training data. The generated data will be saved in `./data_for_train/` (SR_5x5_2x, SR_5x5_4x).**
* **Run `Generate_Data_for_Test.m` to generate test data. The generated data will be saved in `./data_for_test/` (SR_5x5_2x, SR_5x5_4x).**

## Commands for Training
* **Run **`train.py`** to perform network training. Example for training [model_name] on 5x5 angular resolution for 2x/4x SR:**
  ```
  $ python train.py --model_name [model_name] --angRes 5 --scale_factor 2 --batch_size 8
  $ python train.py --model_name [model_name] --angRes 5 --scale_factor 4 --batch_size 4
  ```
* **Checkpoints and Logs will be saved to **`./log/`**, and the **`./log/`** has the following structure:**
  ```
  ├──./log/
  │    ├── SR_5x5_2x
  │    │    ├── [dataset_name]
  │    │         ├── [model_name]
  │    │         │    ├── [model_name]_log.txt
  │    │         │    ├── checkpoints
  │    │         │    │    ├── [model_name]_5x5_2x_epoch_01_model.pth
  │    │         │    │    ├── [model_name]_5x5_2x_epoch_02_model.pth
  │    │         │    │    ├── ...
  │    │         │    ├── results
  │    │         │    │    ├── VAL_epoch_01
  │    │         │    │    ├── VAL_epoch_02
  │    │         │    │    ├── ...
  │    │         ├── [other_model_name]
  │    │         ├── ...
  │    ├── SR_5x5_4x
  ```

## Commands for Test
* **Run **`test.py`** to perform network inference. Example for test [model_name] on 5x5 angular resolution for 2x/4xSR:**
  ```
  $ python test.py --model_name [model_name] --angRes 5 --scale_factor 2  
  $ python test.py --model_name [model_name] --angRes 5 --scale_factor 4 
  ```
  
* **The PSNR and SSIM values of each dataset will be saved to **`./log/`**, and the **`./log/`** has the following structure:**
  ```
  ├──./log/
  │    ├── SR_5x5_2x
  │    │    ├── [dataset_name]
  │    │        ├── [model_name]
  │    │        │    ├── [model_name]_log.txt
  │    │        │    ├── checkpoints
  │    │        │    │   ├── ...
  │    │        │    ├── results
  │    │        │    │    ├── Test
  │    │        │    │    │    ├── evaluation.xls
  │    │        │    │    │    ├── [dataset_1_name]
  │    │        │    │    │    │    ├── [scene_1_name]
  │    │        │    │    │    │    │    ├── [scene_1_name]_CenterView.bmp
  │    │        │    │    │    │    │    ├── [scene_1_name]_SAI.bmp
  │    │        │    │    │    │    │    ├── views
  │    │        │    │    │    │    │    │    ├── [scene_1_name]_0_0.bmp
  │    │        │    │    │    │    │    │    ├── [scene_1_name]_0_1.bmp
  │    │        │    │    │    │    │    │    ├── ...
  │    │        │    │    │    │    │    │    ├── [scene_1_name]_4_4.bmp
  │    │        │    │    │    │    ├── [scene_2_name]
  │    │        │    │    │    │    ├── ...
  │    │        │    │    │    ├── [dataset_2_name]
  │    │        │    │    │    ├── ...
  │    │        │    │    ├── VAL_epoch_01
  │    │        │    │    ├── ...
  │    │        ├── [other_model_name]
  │    │        ├── ...
  │    ├── SR_5x5_4x
  ```



## Benchmark

**We benchmark several methods on the above datasets. PSNR and SSIM metrics are used for quantitative evaluation.
To obtain the metric score for a dataset with `M` scenes, we first calculate the metric on `AxA` SAIs on each scene separately, then obtain the score for each scene by averaging its `A^2` scores, and finally obtain the score for this dataset by averaging the scores of all its `M` scenes.**

**Note: A detailed review of existing LF image SR methods can be referred to [YingqianWang/LF-Image-SR](https://github.com/YingqianWang/LF-Image-SR).**

### PSNR and SSIM values achieved by different methods on 5x5 LFs for 2xSR:
|    Methods    | Scale |  #Params. | EPFL | HCInew | HCIold | INRIA | STFgantry | Model |
| :----------: | :---: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| **Bilinear**     |   x2  | -- | 28.480/0.9180 | 30.718/0.9192 | 36.243/0.9709 | 30.134/0.9455 | 29.577/0.9310 |  |
| **Bicubic**      |   x2  | -- | 29.740/0.9376 | 31.887/0.9356 | 37.686/0.9785 | 31.331/0.9577 | 31.063/0.9498 |
| **VDSR**         |   x2  | 0.665M | 32.498/0.9598 | 34.371/0.9561 | 40.606/0.9867 | 34.439/0.9741 | 35.541/0.9789 |
| **EDSR**         |   x2  | 38.62M | 33.089/0.9629 | 34.828/0.9592 | 41.014/0.9874 | 34.985/0.9764 | 36.296/0.9818 | 
| [**RCAN**](https://github.com/yulunzhang/RCAN)         |   x2  | 15.31M | 33.159/0.9634 | 35.022/0.9603 | 41.125/0.9875 | 35.046/0.9769 | 36.670/0.9831 |
| [**resLF**](https://github.com/shuozh/resLF)        |   x2  | 7.982M | 33.617/0.9706 | 36.685/0.9739 | 43.422/0.9932 | 35.395/0.9804 | 38.354/0.9904 |
| [**LFSSR**](https://github.com/jingjin25/LFSSR-SAS-PyTorch)        |   x2  | 0.888M | 33.671/0.9744 | 36.802/0.9749 | 43.811/0.9938 | 35.279/0.9832 | 37.944/0.9898 |
| [**LF-ATO**](https://github.com/jingjin25/LFSSR-ATO)       |   x2  | 1.216M | 34.272/0.9757 | 37.244/0.9767 | 44.205/0.9942 | 36.170/0.9842 | 39.636/0.9929 |
| [**LF_InterNet**](https://github.com/YingqianWang/LF-InterNet)  |   x2  | 5.040M   | 34.112/0.9760 | 37.170/0.9763 | 44.573/0.9946 | 44.573/0.9946 | 38.435/0.9909 |
| [**LF-DFnet**](https://github.com/YingqianWang/LF-DFnet)     |   x2  |    | 
| [**MEG-Net**](https://github.com/shuozh/MEG-Net)      |   x2  | 1.693M | 34.312/0.9773 | 37.424/0.9777 | 44.097/0.9942 | 36.103/0.9849 | 38.767/0.9915 |
| [**IINet**](https://github.com/GaoshengLiu/LF-IINet)        |   x2  | 4.837M | 34.732/0.9773 | 37.768/0.9790 | 44.852/0.9948 | 36.566/0.9853 | 39.894/0.9936 |
| [**LFT**](https://github.com/ZhengyuLiang24/LFT)          |   x2  | 1.114M | 34.753/0.9778 | 37.762/0.9788 | 44.392/0.9944 | 36.503/0.9854 | 40.316/0.9939 |


### PSNR and SSIM values achieved by different methods on 5x5 angular resolution for 4xSR:

|    Methods    | Scale |  #Params. | EPFL | HCInew | HCIold | INRIA | STFgantry | Model |
| :----------: | :---: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| **Bilinear**     |   x4  | -- | 24.567/0.8158 | 27.085/0.8397 | 31.688/0.9256 | 26.226/0.8757 | 25.203/0.8261 |  |
| **Bicubic**      |   x4  | -- | 25.264/0.8324 | 27.715/0.8517 | 32.576/0.9344 | 26.952/0.8867 | 26.087/0.8452 |  |
| **VDSR**         |   x4  | 0.665M   | 27.246/0.8777 | 29.308/0.8823 | 34.810/0.9515 | 29.186/0.9204|  28.506/0.9009 |
| **EDSR**         |   x4  | 38.89M   | 27.833/0.8854 | 29.591/0.8869 | 35.176/0.9536 | 29.656/0.9257 | 28.703/0.9072 |
| [**RCAN**](https://github.com/yulunzhang/RCAN)         |   x4  | 15.36M | 27.907/0.8863 | 29.694/0.8886 | 35.359/0.9548 | 29.805/0.9276 | 29.021/0.9131 |
| [**resLF**](https://github.com/shuozh/resLF)        |   x4  | 8.646M | 28.260/0.9035 | 30.723/0.9107 | 36.705/0.9682 | 30.338/0.9412 | 30.191/0.9372 |
| [**LFSSR**](https://github.com/jingjin25/LFSSR-SAS-PyTorch)        |   x4  | 1.774M   | 28.596/0.9118 | 30.928/0.9145 | 36.907/0.9696 | 30.585/0.9467 | 30.570/0.9426 |
| [**LF-ATO**](https://github.com/jingjin25/LFSSR-ATO)        |   x4  | 1.364M   | 28.514/0.9115 | 30.880/0.9135 | 36.999/0.9699 | 30.711/0.9484 | 30.607/0.9430 |
| [**LF_InterNet**](https://github.com/YingqianWang/LF-InterNet)  |   x4  | 5.483M   | 28.812/0.9162 | 30.961/0.9161 | 37.150/0.9716 | 30.777/0.9491 | 30.365/0.9409 |
| [**LF-DFnet**](https://github.com/YingqianWang/LF-DFnet)     |   x4  |    | 
| [**MEG-Net**](https://github.com/shuozh/MEG-Net)      |   x4  | 1.775M   | 28.749/0.9160 | 31.103/0.9177 | 37.287/0.9716 | 30.674/0.9490 | 30.771/0.9453 |
| [**LF-IINet**](https://github.com/GaoshengLiu/LF-IINet)        |   x4  | 4.886M   | 29.038/0.9188 | 31.331/0.9208 | 37.620/0.9734 | 31.034/0.9515 | 31.261/0.9502 |
| [**LFT**](https://github.com/ZhengyuLiang24/LFT)           |   x4  | 1.163M | 29.261/0.9209 | 31.433/0.9215 | 37.633/0.9735 | 31.219/0.9524 | 31.795/0.9543 |


## Recources
**We provide the result files generated by the aforementioned methods. (coming soon)**

## To Do List:
* **Upload the result files of each method.

## Acknowledgement

**We would like to thank [Yingqian Wang](https://github.com/YingqianWang) for the helpful discussions and insightful suggestions regarding this repository.**


## Contact
**Welcome to raise issues or email to [zyliang@nudt.edu.cn](zyliang@nudt.edu.cn) for any question regarding our BasicLFSR.**


