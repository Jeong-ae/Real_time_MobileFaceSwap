### [MobileFaceSwap: A Lightweight Framework for Video Face Swapping (AAAI 2022)](https://arxiv.org/abs/2201.03808)
--- 

### Real-time Inference For MobileFaceSwap
- unofficial inference code for real-time MobileFaceSwap using web-cam 

**Dependencies**
For My Environment : ubuntu 20.04 / cuda 11.6 / python 3.8.10 / torch 2.0.0+cu117

```
  python3 -m pip install paddlepaddle-gpu==2.3.1.post116 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
  pip install insightface==0.2.1
  apt-get -y install libgl1-mesa-glx
  pip install protobuf==3.20.*
  pip install onnxruntime==1.14
  pip install Flask
  pip install numpy
  pip install opencv-python

```

**Getting Started**

1. The pretrained models can be downloaded from [Baidu Drive](https://pan.baidu.com/s/14_Wat-OA6ljGfR3Hk8Fk6A) (passward:f6wu) or [Google Drive](https://drive.google.com/file/d/1ZIzGLDB15GRAZAbkfNR0hNWdgQpxeA_r/view?usp=sharing).

2. Run the codes as follows for image or video tests.

```
python image_test.py --target_img_path data/xxx.png --source_img_path data/xxx.png --output_dir results --use_gpu True

python video_test.py --target_video_path data/xxx.mp4 --source_img_path data/xxx.png --output_dir results --use_gpu True
```


**Results**

![](docs/demo.png)

![](docs/video.gif)

**Citation**
```
@inproceedings{xu2022MobileFaceSwap,
  title={MobileFaceSwap: A Lightweight Framework for Video Face Swapping},
  author={Xu, Zhiliang and Hong, Zhibin and Ding, Changxing and Zhu, Zhen and Han, Junyu and Liu, Jingtuo and Ding, Errui},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2022}
}
```
