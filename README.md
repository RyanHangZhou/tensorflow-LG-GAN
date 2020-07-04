# LG-GAN
LG-GAN: Label Guided Adversarial Network for Flexible Targeted Attack of Point Cloud-based Deep Networks
Created by [Hang Zhou](http://home.ustc.edu.cn/~zh2991/), [Dongdong Chen](http://www.dongdongchen.bid/), [Jing Liao](https://liaojing.github.io/html/), [Weiming Zhang](http://staff.ustc.edu.cn/~zhangwm/index.html), [Kejiang Chen](http://home.ustc.edu.cn/~chenkj/), Xiaoyi Dong, Kunlin Liu, [Gang Hua](https://www.ganghua.org/), [Nenghai Yu](http://staff.ustc.edu.cn/~ynh/).

Introduction
--
This repository is for our CVPR 2020 paper '[LG-GAN: Label Guided Adversarial Network for Flexible Targeted Attack of Point Cloud-based Deep Networks](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhou_LG-GAN_Label_Guided_Adversarial_Network_for_Flexible_Targeted_Attack_of_CVPR_2020_paper.html)'. 

Installation
--
This repository is based on Python 3.6, TensorFlow 1.8.0, CUDA 9.0 and cuDNN 7 on Ubuntu 18.04.

1. Set up a virtual environments using conda for the Anaconda Python distribution.

   ```shell
   conda create -n LGGAN python=3.6 anaconda
   ```

2. Install tensorflow-gpu.

   ```shell
 	 pip install tensorflow-gpu==1.8.0
   ```

3. While `nvcc` from CUDA needs to compiling TF operators, install CUDA from CUDA Source Packages. 
   After downloading, implement

   ```shell
   bash cuda_9.0.176_384.81_linux.run --tmpdir=/tmp --override
   ```

   Note that the installation directory is set to `/xxx/cuda-9.0`

4. For compiling TF operators, please check `tf_xxx_compile.sh` under each op subfolder in `tf_ops` folder. Note that you need to update `nvcc`, `python` and `tensoflow` include library if necessary. 

5. Install the other packages

   ```shell
	 pip install h5py
   pip install Pillow
   pip install matplotlib
   ```

6. Point clouds of the ModelNet40 data in HDF5 files are downloaded from  be automatically downloaded (416MB) to the `data` folder. 

  ```shell
	wget -c https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
	unzip modelnet40_ply_hdf5_2048.zip
  ```

7. Download the pretrained PointNet model from [GoogleDrive](https://drive.google.com/drive/folders/11c6v_umZmSHiq-1TLKpSyPQK0E9fDkMU), extract it and put it in folder `checkpoints/pointnet/`. 

8. Download [neural_toolbox](https://github.com/GuessWhatGame/neural_toolbox), extract it and put it in folder `LG-GAN`. 

Usage
--

1. Activate LG-GAN environment.

  ```shell
	source activate LGGAN
  ```

2. Set LD_LIBRARY_PATH.

  ```shell
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/public/zhouhang/cuda-9.0/lib64
  ```

3. Run:

  ```shell
	srun python -u lggan.py --adv_path LGGAN --checkpoints_path LGGAN --log_path LGGAN --tau 1e2
	srun python -u lggan_single.py --adv_path LGGAN_s --checkpoints_path LGGAN_s --log_path LGGAN_s --tau 1e2
	srun python -u lg.py --adv_path LG --checkpoints_path LG --log_path LG --tau 1e2
  ```

Usage
--

If you find our LG-GAN is useful for your research, please consider citing:


@inproceedings{zhou2020lg,
   title={LG-GAN: Label Guided Adversarial Network for Flexible Targeted Attack of Point Cloud-based Deep Networks},
   author={Zhou, Hang and Chen, Dongdong and Liao, Jing and Zhang, Weiming and Dong, Xiaoyi and Liu, Kunlin and Hua, Gang and Yu, Nenghai},
   booktitle={Proceedings of the IEEE International Conference on Computer Vision},
   pages={10356--10365},
   year={2020}
 }
