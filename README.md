# BasicLFSR

### <img src="https://raw.github.com/ZhengyuLiang24/BasicLFSR/main/figs/Thumbnail.jpg" width="1000">

**BasicLFSR is a PyTorch-based open-source and easy-to-use toolbox for Light Field (LF) image Super-Ressolution (SR). This toolbox 
introduces a simple pipeline to train/test your methods, and builds a benchmark to comprehensively evaluate the performance of existing methods.
Our BasicLFSR can help researchers to get access to LF image SR quickly, and facilitates the development of novel methods. Welcome to contribute your own methods to the benchmark.**

**Note: This repository will be updated on a regular basis. Please stay tuned!**

<br>

## Contributions
* **We provide a PyTorch-based open-source and easy-to-use toolbox for LF image SR.**
* **We re-implement a number of existing methods on the unified datasets, and develop a benchmark for performance evaluation.**
* **We share the codes, models and results of existing methods to help researchers better get access to this area.**

<br>

## News & Updates
* **Jan 18, 2022: Upload the result files of [DPT](https://github.com/BITszwang/DPT), accepted to AAAI 2022.**
* **Jan 02, 2022: Upload the result files of existing methods to OneDrive.**
* **Des 12, 2021: Add the dataset generation codes based on Python, i.e., `Generate_Data_for_Training.py` and `Generate_Data_for_Test.py`.**
* **Dec 10, 2021: Add the comparisions of model sizes of existing methods.**

<br>

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
* **Run `Generate_Data_for_Training.m` or `Generate_Data_for_Training.py` to generate training data. The generated data will be saved in `./data_for_train/` (SR_5x5_2x, SR_5x5_4x).**
* **Run `Generate_Data_for_Test.m` or `Generate_Data_for_Test.py` to generate test data. The generated data will be saved in `./data_for_test/` (SR_5x5_2x, SR_5x5_4x).**

<br>

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

<br>

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

<br>

## Benchmark

**We benchmark several methods on the above datasets. PSNR and SSIM metrics are used for quantitative evaluation.
To obtain the metric score for a dataset with `M` scenes, we first calculate the metric on `AxA` SAIs on each scene separately, then obtain the score for each scene by averaging its `A^2` scores, and finally obtain the score for this dataset by averaging the scores of all its `M` scenes.**

**Note: A detailed review of existing LF image SR methods can be referred to [YingqianWang/LF-Image-SR](https://github.com/YingqianWang/LF-Image-SR).**

### PSNR and SSIM values achieved by different methods on 5x5 LFs for 2xSR:
|    Methods    | Scale |  #Params. | EPFL | HCInew | HCIold | INRIA | STFgantry | Results |
| :----------: | :---: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| **Bilinear**     |   x2  |     -- | 28.480/0.9180 | 30.718/0.9192 | 36.243/0.9709 | 30.134/0.9455 | 29.577/0.9310 | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/zyliang_stu_xidian_edu_cn/ES52EP0x8jlNsqXMl6scfAYBqOzQyzjo08wyXlqBAV-eXg?e=YnJAn1) |
| **Bicubic**      |   x2  |     -- | 29.740/0.9376 | 31.887/0.9356 | 37.686/0.9785 | 31.331/0.9577 | 31.063/0.9498 | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/zyliang_stu_xidian_edu_cn/ESRqpmDMyGZOleXrhS01ZlgBwPMclcakv6wsbAkEBU_qWA?e=d9EFMu) |
| **VDSR**         |   x2  | 0.665M | 32.498/0.9598 | 34.371/0.9561 | 40.606/0.9867 | 34.439/0.9741 | 35.541/0.9789 | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/zyliang_stu_xidian_edu_cn/EdvzztunSM1Ek3nZqWYdO2IBTAR-3lCuZSuRL0xP48i1KQ?e=qcJvP7) |
| **EDSR**         |   x2  | 38.62M | 33.089/0.9629 | 34.828/0.9592 | 41.014/0.9874 | 34.985/0.9764 | 36.296/0.9818 | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/zyliang_stu_xidian_edu_cn/EX1nr44rinlFoGNjQ5n2HjkBPuG50Y4js1wLejPTYV2QFA?e=MXwm2y) |
| [**RCAN**](https://github.com/yulunzhang/RCAN)                 |   x2  | 15.31M | 33.159/0.9634 | 35.022/0.9603 | 41.125/0.9875 | 35.046/0.9769 | 36.670/0.9831 | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/zyliang_stu_xidian_edu_cn/EXjwLdjBqHlCmko76WMcwQcBE3SYeCdmsDvB1D0pUAD4Bg?e=h4KVQu) |
| [**resLF**](https://github.com/shuozh/resLF)                   |   x2  | 7.982M | 33.617/0.9706 | 36.685/0.9739 | 43.422/0.9932 | 35.395/0.9804 | 38.354/0.9904 | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/zyliang_stu_xidian_edu_cn/EZlDlA5vdSlGh795F63t12MBrlEXG8ufKO91HOLOIaELYA?e=x8ki4d) |
| [**LFSSR**](https://github.com/jingjin25/LFSSR-SAS-PyTorch)    |   x2  | 0.888M | 33.671/0.9744 | 36.802/0.9749 | 43.811/0.9938 | 35.279/0.9832 | 37.944/0.9898 | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/zyliang_stu_xidian_edu_cn/EcgmEOxh02BLt2HDBaRBq3oBG8zkmgGqhIXjmJepzpQofQ?e=eecUwk) |
| [**LF-ATO**](https://github.com/jingjin25/LFSSR-ATO)           |   x2  | 1.216M | 34.272/0.9757 | 37.244/0.9767 | 44.205/0.9942 | 36.170/0.9842 | 39.636/0.9929 | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/zyliang_stu_xidian_edu_cn/EUrVxJstMMZHoZCEM4ZZlNABHOQMPV3l5wnz4vO9DJ5lgg?e=ZBBKjv) |
| [**LF_InterNet**](https://github.com/YingqianWang/LF-InterNet) |   x2  | 5.040M | 34.112/0.9760 | 37.170/0.9763 | 44.573/0.9946 | 35.829/0.9843 | 38.435/0.9909 | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/zyliang_stu_xidian_edu_cn/EaC5TT7K5dFJnN1KwRtLkzUBI876TFd0ZKQqeJR3I4tMfg?e=PrCRFg) |
| [**LF-DFnet**](https://github.com/YingqianWang/LF-DFnet)       |   x2  | 3.940M | 34.513/0.9755 | 37.418/0.9773 | 44.198/0.9941 | 36.416/0.9840 | 39.427/0.9926 | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/zyliang_stu_xidian_edu_cn/EXRgsDklzLVFj-e0jqlzppoBFoHkl4vtBWHis58q6uyd4w?e=k71q0a) |
| [**MEG-Net**](https://github.com/shuozh/MEG-Net)               |   x2  | 1.693M | 34.312/0.9773 | 37.424/0.9777 | 44.097/0.9942 | 36.103/0.9849 | 38.767/0.9915 | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/zyliang_stu_xidian_edu_cn/ERTfTtTrxZFKg_aFWDY4UPoBwarWfb4-N6tWFnyGFO1B_g?e=3Es5TK) |
| [**LF-IINet**](https://github.com/GaoshengLiu/LF-IINet)        |   x2  | 4.837M | 34.732/0.9773 | 37.768/0.9790 | 44.852/0.9948 | 36.566/0.9853 | 39.894/0.9936 | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/zyliang_stu_xidian_edu_cn/ET2p0yPcHwhMlgEVseaIzqoBYVxcSWbErtUt31ExtbYkDw?e=Kz0gTW) |
| [**LFT**](https://github.com/ZhengyuLiang24/LFT)               |   x2  | 1.114M | 34.753/0.9778 | 37.762/0.9788 | 44.392/0.9944 | 36.503/0.9854 | 40.316/0.9939 | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/zyliang_stu_xidian_edu_cn/ETVEt-4UanxApjXy8e9PDG8BvGE9RQ7Gc7Xl3QsV_FClLQ?e=nFAnLi) |
| [**DPT**](https://github.com/BITszwang/DPT)                    |   x2  | 3.731M | 34.490/0.9758 | 37.355/0.9771 | 44.302/0.9943 | 36.409/0.9843 | 39.429/0.9926 | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/zyliang_stu_xidian_edu_cn/EYYILn3mmSVAtwAkk90fHs8B7tjPIQJQLXLTNhVohP_8Ng?e=lZHFqe)

### PSNR and SSIM values achieved by different methods on 5x5 angular resolution for 4xSR:

|    Methods    | Scale |  #Params. | EPFL | HCInew | HCIold | INRIA | STFgantry | Results |
| :----------: | :---: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| **Bilinear**     |   x4  |     -- | 24.567/0.8158 | 27.085/0.8397 | 31.688/0.9256 | 26.226/0.8757 | 25.203/0.8261 | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/zyliang_stu_xidian_edu_cn/EYYAOUOHEUFDuJOGBtSKhm4B-XkAf_GA3TakJPqs1qZmqQ?e=BfPH5x) |
| **Bicubic**      |   x4  |     -- | 25.264/0.8324 | 27.715/0.8517 | 32.576/0.9344 | 26.952/0.8867 | 26.087/0.8452 | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/zyliang_stu_xidian_edu_cn/EcYP3g5wNnFJlSNq4fOzojEBPubKqhJSL7jQIOgEITepjg?e=aLYa3T) |
| **VDSR**         |   x4  | 0.665M | 27.246/0.8777 | 29.308/0.8823 | 34.810/0.9515 | 29.186/0.9204|  28.506/0.9009 | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/zyliang_stu_xidian_edu_cn/EULGAsGxYIBOoqkCPWyWb54BnRwm4Y9YsYFosucKseJhPg?e=SajUcg) |
| **EDSR**         |   x4  | 38.89M | 27.833/0.8854 | 29.591/0.8869 | 35.176/0.9536 | 29.656/0.9257 | 28.703/0.9072 | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/zyliang_stu_xidian_edu_cn/EaTe1CYuGQVDor_FpMHpmecBGY-1JMj2rVOboQAT_5BHyA?e=MOg5Ww) |
| [**RCAN**](https://github.com/yulunzhang/RCAN)                  |   x4  | 15.36M | 27.907/0.8863 | 29.694/0.8886 | 35.359/0.9548 | 29.805/0.9276 | 29.021/0.9131 | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/zyliang_stu_xidian_edu_cn/Ed4kCb434FdOgjZ7xwuz-QcBRuJ4DzQVVStYOkS-D54Wxg?e=2q6L1s) |
| [**resLF**](https://github.com/shuozh/resLF)                    |   x4  | 8.646M | 28.260/0.9035 | 30.723/0.9107 | 36.705/0.9682 | 30.338/0.9412 | 30.191/0.9372 | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/zyliang_stu_xidian_edu_cn/EX-1RIiv5YdLpuHcRGWBuGMB4pZbvgGdjcymMMmOahGAgw?e=aaPSCv) |
| [**LFSSR**](https://github.com/jingjin25/LFSSR-SAS-PyTorch)     |   x4  | 1.774M | 28.596/0.9118 | 30.928/0.9145 | 36.907/0.9696 | 30.585/0.9467 | 30.570/0.9426 | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/zyliang_stu_xidian_edu_cn/EXVy3Ty71opAkAqIFOjWuNgBs-H-GNRzFQfRKA_yBVsgfA?e=yjMFT5) |
| [**LF-ATO**](https://github.com/jingjin25/LFSSR-ATO)            |   x4  | 1.364M | 28.514/0.9115 | 30.880/0.9135 | 36.999/0.9699 | 30.711/0.9484 | 30.607/0.9430 | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/zyliang_stu_xidian_edu_cn/EdHXXqAQiOFAmFb2QyIpsCYBsfrmpW46RUiQJw7MgozddA?e=Cla1yU) |
| [**LF_InterNet**](https://github.com/YingqianWang/LF-InterNet)  |   x4  | 5.483M | 28.812/0.9162 | 30.961/0.9161 | 37.150/0.9716 | 30.777/0.9491 | 30.365/0.9409 | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/zyliang_stu_xidian_edu_cn/EeFD_x_wOh5JnE9kkv-Qz-UBFYYp1O-cjAPpRUUGFWVCSQ?e=f65RUq) |
| [**LF-DFnet**](https://github.com/YingqianWang/LF-DFnet)        |   x4  | 3.990M | 28.774/0.9165 | 31.234/0.9196 | 37.321/0.9718 | 30.826/0.9503 | 31.147/0.9494 | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/zyliang_stu_xidian_edu_cn/EVSo-PNpvetCqZUHoaTY81ABpzBbW19ApDLr-8NW1Eq81g?e=whIEAy) |
| [**MEG-Net**](https://github.com/shuozh/MEG-Net)                |   x4  | 1.775M | 28.749/0.9160 | 31.103/0.9177 | 37.287/0.9716 | 30.674/0.9490 | 30.771/0.9453 | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/zyliang_stu_xidian_edu_cn/Ea76GJHEcp5NmysacM-MLXUBLaKsqT3LseZgrjM2-_QgiA?e=hJSmOQ) |
| [**LF-IINet**](https://github.com/GaoshengLiu/LF-IINet)         |   x4  | 4.886M | 29.038/0.9188 | 31.331/0.9208 | 37.620/0.9734 | 31.034/0.9515 | 31.261/0.9502 | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/zyliang_stu_xidian_edu_cn/EdsomADGXzhPlNk9FDolGI8BHaj_JpBAutibC1GOhRx5Dg?e=mlPUWk) |
| [**LFT**](https://github.com/ZhengyuLiang24/LFT)                |   x4  | 1.163M | 29.261/0.9209 | 31.433/0.9215 | 37.633/0.9735 | 31.219/0.9524 | 31.795/0.9543 | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/zyliang_stu_xidian_edu_cn/EWAD9t3MAuVMnJFF7U_2UlwBs2KjabTmdR9qlpKbt3btSA?e=Be8lLe) |
| [**DPT**](https://github.com/BITszwang/DPT)                     |   x4  | 3.778M | 28.939/0.9170 | 31.196/0.9188 | 37.412/0.9721 | 30.964/0.9503 | 31.150/0.9488 | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/zyliang_stu_xidian_edu_cn/EW7MkV9H97FOjEqb1j1NiMMBaNvm5-zdXYhQ3531HeLozg?e=ARYqTe)
<br>

## Recources
**We provide the result files generated by the aforementioned methods, and researchers can download the results via [this link](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/zyliang_stu_xidian_edu_cn/Emdf-dQmFtxBuezIoItaQI4BQA0v3yC-6X8cj5pNyDqm-A?e=OOLEIe).**

<br>

## To Do List:


<br>

## Acknowledgement

**We would like to thank [Yingqian Wang](https://github.com/YingqianWang) for the helpful discussions and insightful suggestions regarding this repository.**


## Contact
**Welcome to raise issues or email to [zyliang@nudt.edu.cn](zyliang@nudt.edu.cn) for any question regarding our BasicLFSR.**


