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
* **Jul 14, 2023: [EPIT](https://github.com/ZhengyuLiang24/EPIT) is accepted to ICCV 2023.**
* **Jul 01, 2023: Update the pre-trained models of benchmark methods to [Releases](https://github.com/ZhengyuLiang24/BasicLFSR/releases).**
* **Jul 01, 2023: Add a new work [LF-DET](https://github.com/Congrx/LF-DET), accepted to IEEE TMM.**
* **Mar 31, 2023: Add a new benchmark [NTIRE-2023](https://codalab.lisn.upsaclay.fr/competitions/9201).**
* **Mar 29, 2023: Add a new work [HLFSR-SSR](https://github.com/duongvinh/HLFSR-SSR), accepted to IEEE TCI.**
* **Feb 16, 2023: Add a new work [EPIT](https://github.com/ZhengyuLiang24/EPIT).**
* **Dec 10, 2022: Add a new work [LFSSR_SAV](https://github.com/Joechann0831/SAV_conv), accepted to IEEE TCI.**



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
|                            Methods                             | Scale | #Params. |         EPFL          |       HCInew        |       HCIold        |         INRIA         |       STFgantry       |
|:--------------------------------------------------------------:|:-----:|:--------:|:---------------------:|:-------------------:|:-------------------:|:---------------------:|:---------------------:|
|                          **Bilinear**                          |  x2   |    --    |     28.480/0.9180     |    30.718/0.9192    |    36.243/0.9709    |     30.134/0.9455     |     29.577/0.9310     |
|                          **Bicubic**                           |  x2   |    --    |     29.740/0.9376     |    31.887/0.9356    |    37.686/0.9785    |     31.331/0.9577     |     31.063/0.9498     |
|                            **VDSR**                            |  x2   |  0.665M  |     32.498/0.9598     |    34.371/0.9561    |    40.606/0.9867    |     34.439/0.9741     |     35.541/0.9789     |
|                            **EDSR**                            |  x2   |  38.62M  |     33.089/0.9629     |    34.828/0.9592    |    41.014/0.9874    |     34.985/0.9764     |     36.296/0.9818     |
|         [**RCAN**](https://github.com/yulunzhang/RCAN)         |  x2   |  15.31M  |     33.159/0.9634     |    35.022/0.9603    |    41.125/0.9875    |     35.046/0.9769     |     36.670/0.9831     |
|          [**resLF**](https://github.com/shuozh/resLF)          |  x2   |  7.982M  |     33.617/0.9706     |    36.685/0.9739    |    43.422/0.9932    |     35.395/0.9804     |     38.354/0.9904     |
|  [**LFSSR**](https://github.com/jingjin25/LFSSR-SAS-PyTorch)   |  x2   |  0.888M  |     33.671/0.9744     |    36.802/0.9749    |    43.811/0.9938    |     35.279/0.9832     |     37.944/0.9898     |
|      [**LF-ATO**](https://github.com/jingjin25/LFSSR-ATO)      |  x2   |  1.216M  |     34.272/0.9757     |    37.244/0.9767    |    44.205/0.9942    |     36.170/0.9842     |     39.636/0.9929     |
| [**LF_InterNet**](https://github.com/YingqianWang/LF-InterNet) |  x2   |  5.040M  |     34.112/0.9760     |    37.170/0.9763    |    44.573/0.9946    |     35.829/0.9843     |     38.435/0.9909     |
|    [**LF-DFnet**](https://github.com/YingqianWang/LF-DFnet)    |  x2   |  3.940M  |     34.513/0.9755     |    37.418/0.9773    |    44.198/0.9941    |     36.416/0.9840     |     39.427/0.9926     |
|        [**MEG-Net**](https://github.com/shuozh/MEG-Net)        |  x2   |  1.693M  |     34.312/0.9773     |    37.424/0.9777    |    44.097/0.9942    |     36.103/0.9849     |     38.767/0.9915     |
|    [**LF-IINet**](https://github.com/GaoshengLiu/LF-IINet)     |  x2   |  4.837M  |     34.732/0.9773     |    37.768/0.9790    |    44.852/0.9948    |     36.566/0.9853     |     39.894/0.9936     |
|          [**DPT**](https://github.com/BITszwang/DPT)           |  x2   |  3.731M  |     34.490/0.9758     |    37.355/0.9771    |    44.302/0.9943    |     36.409/0.9843     |     39.429/0.9926     |
|        [**LFT**](https://github.com/ZhengyuLiang24/LFT)        |  x2   |  1.114M  |     34.804/0.9781     |    37.838/0.9791    |    44.522/0.9945    |     36.594/0.9855     |     40.510/0.9941     | 
|    [**DistgSSR**](https://github.com/YingqianWang/DistgSSR)    |  x2   |  3.532M  |     34.809/0.9787     |    37.959/0.9796    |    44.943/0.9949    |     36.586/0.9859     |     40.404/0.9942     |
|   [**LFSSR_SAV**](https://github.com/Joechann0831/SAV_conv)    |  x2   |  1.217M  |     34.616/0.9772     |    37.425/0.9776    |    44.216/0.9942    |     36.364/0.9849     |     38.689/0.9914     |
|       [**EPIT**](https://github.com/ZhengyuLiang24/EPIT)       |  x2   |  1.421M  |     34.826/0.9775     |  38.228/**0.9810**  |  **45.075**/0.9949  |     36.672/0.9853     | **42.166**/**0.9957** |
|    [**HLFSR-SSR**](https://github.com/duongvinh/HLFSR-SSR)     |  x2   |  13.72M  | **35.310**/**0.9800** | **38.317**/*0.9807* |  44.978/**0.9950**  | **37.060**/**0.9867** |     40.849/0.9947     |
|         [**LF-DET**](https://github.com/Congrx/LF-DET)         |  x2   |  1.588M  |   *35.262*/*0.9797*   |  *38.314*/*0.9807*  | *44.986*/**0.9950** |   *36.949*/*0.9864*   |   *41.762*/*0.9955*   |
<br>

### PSNR and SSIM values achieved by different methods on 5x5 angular resolution for 4xSR:

|                            Methods                             | Scale | #Params. |         EPFL          |        HCInew         |        HCIold         |         INRIA         |      STFgantry      |
|:--------------------------------------------------------------:|:-----:|:--------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:-------------------:|
|                          **Bilinear**                          |  x4   |    --    |     24.567/0.8158     |     27.085/0.8397     |     31.688/0.9256     |     26.226/0.8757     |    25.203/0.8261    |
|                          **Bicubic**                           |  x4   |    --    |     25.264/0.8324     |     27.715/0.8517     |     32.576/0.9344     |     26.952/0.8867     |    26.087/0.8452    | 
|                            **VDSR**                            |  x4   |  0.665M  |     27.246/0.8777     |     29.308/0.8823     |     34.810/0.9515     |     29.186/0.9204     |    28.506/0.9009    | 
|                            **EDSR**                            |  x4   |  38.89M  |     27.833/0.8854     |     29.591/0.8869     |     35.176/0.9536     |     29.656/0.9257     |    28.703/0.9072    |
|         [**RCAN**](https://github.com/yulunzhang/RCAN)         |  x4   |  15.36M  |     27.907/0.8863     |     29.694/0.8886     |     35.359/0.9548     |     29.805/0.9276     |    29.021/0.9131    |
|          [**resLF**](https://github.com/shuozh/resLF)          |  x4   |  8.646M  |     28.260/0.9035     |     30.723/0.9107     |     36.705/0.9682     |     30.338/0.9412     |    30.191/0.9372    |
|  [**LFSSR**](https://github.com/jingjin25/LFSSR-SAS-PyTorch)   |  x4   |  1.774M  |     28.596/0.9118     |     30.928/0.9145     |     36.907/0.9696     |     30.585/0.9467     |    30.570/0.9426    |
|      [**LF-ATO**](https://github.com/jingjin25/LFSSR-ATO)      |  x4   |  1.364M  |     28.514/0.9115     |     30.880/0.9135     |     36.999/0.9699     |     30.711/0.9484     |    30.607/0.9430    |
| [**LF_InterNet**](https://github.com/YingqianWang/LF-InterNet) |  x4   |  5.483M  |     28.812/0.9162     |     30.961/0.9161     |     37.150/0.9716     |     30.777/0.9491     |    30.365/0.9409    | 
|    [**LF-DFnet**](https://github.com/YingqianWang/LF-DFnet)    |  x4   |  3.990M  |     28.774/0.9165     |     31.234/0.9196     |     37.321/0.9718     |     30.826/0.9503     |    31.147/0.9494    | 
|        [**MEG-Net**](https://github.com/shuozh/MEG-Net)        |  x4   |  1.775M  |     28.749/0.9160     |     31.103/0.9177     |     37.287/0.9716     |     30.674/0.9490     |    30.771/0.9453    |
|    [**LF-IINet**](https://github.com/GaoshengLiu/LF-IINet)     |  x4   |  4.886M  |     29.038/0.9188     |     31.331/0.9208     |     37.620/0.9734     |     31.034/0.9515     |    31.261/0.9502    |
|          [**DPT**](https://github.com/BITszwang/DPT)           |  x4   |  3.778M  |     28.939/0.9170     |     31.196/0.9188     |     37.412/0.9721     |     30.964/0.9503     |    31.150/0.9488    |
|        [**LFT**](https://github.com/ZhengyuLiang24/LFT)        |  x4   |  1.163M  |     29.255/0.9210     |     31.462/0.9218     |     37.630/0.9735     |     31.205/0.9524     |    31.860/0.9548    |
|    [**DistgSSR**](https://github.com/YingqianWang/DistgSSR)    |  x4   |  3.582M  |     28.992/0.9195     |     31.380/0.9217     |     37.563/0.9732     |     30.994/0.9519     |    31.649/0.9535    |
|   [**LFSSR_SAV**](https://github.com/Joechann0831/SAV_conv)    |  x4   |  1.543M  |   *29.368*/*0.9223*   |     31.450/0.9217     |     37.497/0.9721     |     31.270/0.9531     |    31.362/0.9505    |
|       [**EPIT**](https://github.com/ZhengyuLiang24/EPIT)       |  x4   |  1.470M  |     29.339/0.9197     |     31.511/0.9231     |     37.677/0.9737     |    *31.372*/0.9526    | **32.179**/*0.9571* |
|    [**HLFSR-SSR**](https://github.com/duongvinh/HLFSR-SSR)     |  x4   |  13.87M  |     29.196/0.9222     | **31.571**/**0.9238** |   *37.776*/*0.9742*   |   31.241/**0.9534**   |    31.641/0.9537    |
|         [**LF-DET**](https://github.com/Congrx/LF-DET)         |  x4   |  1.687M  | **29.473**/**0.9230** |   *31.558*/*0.9235*   | **37.843**/**0.9744** | **31.389**/**0.9534** | *32.139*/**0.9573** |
<br>

**We provide the result files generated by the aforementioned methods, and researchers can download the results via [this link](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/zyliang_stu_xidian_edu_cn/Emdf-dQmFtxBuezIoItaQI4BQA0v3yC-6X8cj5pNyDqm-A?e=OOLEIe).**

<br>
<br>

## NTIRE 2023 LF Image SR Challenge
### <img src="https://raw.github.com/ZhengyuLiang24/BasicLFSR/main/figs/NTIRE2023.png" width="1000">
**[NTIRE 2023 LFSR Challenge](https://github.com/The-Learning-And-Vision-Atelier-LAVA/LF-Image-SR/tree/NTIRE2023) introduces a new LF dataset (namely, NTIRE-2023) for validation and test. Both the validation and testset sets contain 16 synthetic scenes rendered by the [3DS MAX software](https://www.autodesk.eu/products/3ds-max/overview) and 16 real-world images captured by Lytro Illum cameras. For synthetic scenes, all virtual cameras in the camera array have identical internal parameters and are co-planar with the parallel optical axes.**

**All scenes in the test set have an angular resolution of $5\times 5$. The spatial resolutions of synthetic LFs and real-world LFs are $500\times500$ and $624\times432$, respectively. All the LF images in the test set are bicubicly downsampled by a factor of $4$. The participants are required to apply their models to the LR LF images released via [OneDrive](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/zyliang_stu_xidian_edu_cn/EiEJwlGY3SZDi0FMvHsIMUMB2c73kFsAqELkeidVGoOsKA?e=xHhOEG), and submit their $4\times$ super-resolved LF images to the [CodaLab platform](https://codalab.lisn.upsaclay.fr/competitions/9201#results) for test.**

**Only the LR versions are released to the participants, and the test server is still online.**

### Benchmark on NTIRE-2023 Test

|                            Methods                             | Scale | #Params. |        Lytro        |      Synthetic      |       Average       |
|:--------------------------------------------------------------:|:-----:|:--------:|:-------------------:|:-------------------:|:-------------------:|
|                          **Bicubic**                           |  x4   |    --    |    25.109/0.8404    |    26.461/0.8352    |    25.785/0.8378    |
|                            **VDSR**                            |  x4   |  0.665M  |    27.052/0.8888    |    27.936/0.8703    |    27.494/0.8795    |
|                            **EDSR**                            |  x4   |  38.89M  |    27.540/0.8981    |    28.206/0.8757    |    27.873/0.8869    |
|         [**RCAN**](https://github.com/yulunzhang/RCAN)         |  x4   |  15.36M  |    27.606/0.9001    |    28.308/0.8773    |    27.957/0.8887    |
|          [**resLF**](https://github.com/shuozh/resLF)          |  x4   |  8.646M  |    28.657/0.9260    |    29.245/0.8968    |    28.951/0.9114    |
|  [**LFSSR**](https://github.com/jingjin25/LFSSR-SAS-PyTorch)   |  x4   |  1.774M  |    29.029/0.9337    |    29.399/0.9008    |    29.214/0.9173    |
|      [**LF-ATO**](https://github.com/jingjin25/LFSSR-ATO)      |  x4   |  1.364M  |    29.087/0.9354    |    29.401/0.9012    |    29.244/0.9183    |
| [**LF_InterNet**](https://github.com/YingqianWang/LF-InterNet) |  x4   |  5.483M  |    29.233/0.9369    |    29.446/0.9028    |    29.340/0.9198    |
|        [**MEG-Net**](https://github.com/shuozh/MEG-Net)        |  x4   |  1.775M  |    29.203/0.9369    |    29.539/0.9036    |    29.371/0.9203    |
|    [**LF-IINet**](https://github.com/GaoshengLiu/LF-IINet)     |  x4   |  4.886M  |    29.487/0.9403    |    29.786/0.9071    |    29.636/0.9237    |
|          [**DPT**](https://github.com/BITszwang/DPT)           |  x4   |  3.778M  |    29.360/0.9388    |    29.771/0.9064    |    29.566/0.9226    |
|        [**LFT**](https://github.com/ZhengyuLiang24/LFT)        |  x4   |  1.163M  |    29.657/0.9420    |    29.881/0.9084    |    29.769/0.9252    |
|    [**DistgSSR**](https://github.com/YingqianWang/DistgSSR)    |  x4   |  3.582M  |    29.389/0.9403    |    29.884/0.9084    |    29.637/0.9244    |
|   [**LFSSR_SAV**](https://github.com/Joechann0831/SAV_conv)    |  x4   |  1.543M  |  29.713/**0.9425**  |    29.850/0.9075    |    29.782/0.9250    |
|       [**EPIT**](https://github.com/ZhengyuLiang24/EPIT)       |  x4   |  1.470M  |  *29.718*/*0.9420*  | **30.030**/*0.9097* |   *29.874*/0.9259   |
|    [**HLFSR-SSR**](https://github.com/duongvinh/HLFSR-SSR)     |  x4   |  13.87M  |    29.714/0.9429    |   29.945/*0.9097*   |  29.830/**0.9263**  |
|         [**LF-DET**](https://github.com/Congrx/LF-DET)         |  x4   |  1.687M  | **29.911**/*0.9420* | *29.976*/**0.9101** | **29.944**/*0.9260* |

**We provide the result files generated by the aforementioned methods, and researchers can download the results via [this link](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/zyliang_stu_xidian_edu_cn/EsiZrL7YPrBJpzf9uOdPrvsBYxM-sVJXJo6xBoDQlwxQwg?e=JiPjIN).**

<br>
<br>


## Citiation
```
@InProceedings{NTIRE2023LFSR,
  author    = {Wang, Yingqian and Wang, Longguang and Liang, Zhengyu and Yang, Jungang and Timofte, Radu and Guo, Yulan and Jin, Kai and Wei, Zeqiang and Yang, Angulia and Guo, Sha and Gao, Mingzhi and Zhou, Xiuzhuang and Duong, Vinh Van and Huu, Thuc Nguyen and Yim, Jonghoon and Jeon, Byeungwoo and Liu, Yutong and Cheng, Zhen and Xiao, Zeyu and Xu, Ruikang and Xiong, Zhiwei and Liu, Gaosheng and Jin, Manchang and Yue, Huanjing and Yang, Jingyu and Gao, Chen and Zhang, Shuo and Chang, Song and Lin, Youfang and Chao, Wentao and Wang, Xuechun and Wang, Guanghui and Duan, Fuqing and Xia, Wang and Wang, Yan and Xia, Peiqi and Wang, Shunzhou and Lu, Yao and Cong, Ruixuan and Sheng, Hao and Yang, Da and Chen, Rongshan and Wang, Sizhe and Cui, Zhenglong and Chen, Yilei and Lu, Yongjie and Cai, Dongjun and An, Ping and Salem, Ahmed and Ibrahem, Hatem and Yagoub, Bilel and Kang, Hyun-Soo and Zeng, Zekai and Wu, Heng},
  title     = {NTIRE 2023 Challenge on Light Field Image Super-Resolution: Dataset, Methods and Results},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year      = {2023},
}
```


<br>

## Resources
* **The pre-trained models of the aforementioned methods can be downloaded via [Releases](https://github.com/ZhengyuLiang24/BasicLFSR/releases).**

<br>

## Acknowledgement

**We thank [Yingqian Wang](https://github.com/YingqianWang) for the helpful discussions and insightful suggestions regarding this repository.**


## Contact
**Welcome to raise issues or email to [zyliang@nudt.edu.cn](zyliang@nudt.edu.cn) for any question regarding our BasicLFSR.**


