```yaml
title: SO-VITS-SVC 4.0 全流程详解
tag: SO-VITS-SVC
```

# SO-VITS-SVC 4.0 全流程详解

本篇目为项目[SO-VITS-SVC 4.0](https://github.com/svc-develop-team/so-vits-svc)的预处理及训练、推理提供相关教程，建议用于参考而非完全使用。~~本文章不是生活必需品~~，您也可以直接使用[官方文档](https://github.com/svc-develop-team/so-vits-svc/blob/4.0/README_zh_CN.md)

---

## 法律条例参考及免责声明

**《民法典》**

**第一千零一十九条**

任何组织或者个人不得以丑化、污损，或者利用信息技术手段伪造等方式侵害他人的肖像权。未经肖像权人同意，不得制作、使用、公开肖像权人的肖像，但是法律另有规定的除外。 未经肖像权人同意，肖像作品权利人不得以发表、复制、发行、出租、展览等方式使用或者公开肖像权人的肖像。 对自然人声音的保护，参照适用肖像权保护的有关规定。

**第一千零二十四条**

【名誉权】民事主体享有名誉权。任何组织或者个人不得以侮辱、诽谤等方式侵害他人的名誉权。

**第一千零二十七条**

【作品侵害名誉权】行为人发表的文学、艺术作品以真人真事或者特定人为描述对象，含有侮辱、诽谤内容，侵害他人名誉权的，受害人有权依法请求该行为人承担民事责任。 行为人发表的文学、艺术作品不以特定人为描述对象，仅其中的情节与该特定人的情况相似的，不承担民事责任。   

**<u>请自行解决数据集授权问题，禁止使用非授权数据集进行训练！任何由于使用非授权数据集进行训练造成的问题，需自行承担全部责任和后果！与仓库、仓库维护者、svc develop team 、本教程撰写者无关！</u>**

---

## 1.环境依赖

> Python <= 3.10
> 
> NVIDIA-CUDA <= 11.8
> 
> Pytorch <= 2.0
> FFmpeg
> 
> *依赖库*

### Python

前往[Python官网](https://www.python.org/)下载Python，版本需要低于3.10，目前暂不支持Python3.11。

安装完成后请输入`python -V`确认所安装的python版本。

如果你实在不会，可参阅此文章：[Python 环境搭建 | 菜鸟教程](https://www.runoob.com/python/python-install.html)

### CUDA

在控制台中输入`nvidia-smi` 查看显卡驱动并检查所安装的CUDA版本。

若你未安装CUDA，请前往[这里](https://developer.nvidia.com/cuda-11-7-0-download-archive)选择你所使用机器的对应版本下载安装。

### Pytorch

安装Pytorch请注意所安装的CUDA版本，在[Pytorch安装页](https://pytorch.org/get-started/locally/)选择对应的Pytorch版本，不同CUDA版本的安装命令有所不同。**若版本不正确会导致Pytorch不能正常运行。** 运行命令后将会自动安装`torch` `torchvision` `torchaudio`三个pypi包。

<img src="file:///C:/Users/lemon/AppData/Roaming/marktext/images/2023-03-19-21-05-59-image.png" title="" alt="" data-align="center">

运行命令安装后请完成以下步骤检查Pytorch的运行情况

1. 在控制台中输入`python`进入交互式解释器。

2. 输入以下命令并检查返回结果
   
   ```shell
   >>>import torch    # 如正常则静默
   >>>torch.cuda.is_available()
   True
   ```

若返回结果不为`True`而为`False`，则需要重新安装直至返回值为`True`以表示目前GPU已可被正常使用。

`False`表示Pytorch无法正常调用GPU。

如果你所使用的环境曾经使用过其他使用了Pytorch的项目，请注意Pytorch版本与例如Xformers等的其他程序是否冲突。

### FFmpeg

前往[FFmpeg官网](https://ffmpeg.org/)下载并解压，然后前往<u>系统环境变量</u>并在Path中添加FFmpeg解压后bin目录的位置[`.\ffmpeg\bin`]，完成后在控制台中输入ffmpeg确认是否有输出。

## 2.训练

### 2.1 获取源码

前往[so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)项目仓库**4.0**分支获取源码，或直接clone以下地址获取主要分支源码

```shell
git@github.com:svc-develop-team/so-vits-svc.git
```

### 2.2 获取预先下载的模型文件

#### 必须项

- contentvec ：[checkpoint_best_legacy_500.pt](https://ibm.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr)
  - 放在`hubert`目录下

#### 可选项（建议使用）

- 预训练底模文件： `G_0.pth` `D_0.pth`
  - 放在`logs/44k`目录下

目前`G_0.pth` 及 `D_0.pth` （底模）没有官方获得途径，请自行寻找获取方式。

本文章撰写者在HuggingFace上找到了个不知道是什么的链接，可以看看，里面是什么我真的不知道.jpg [justinjohn-03/so-vits-svc-4.0 at main](https://huggingface.co/justinjohn-03/so-vits-svc-4.0/tree/main)

*虽然底模一般不会引起什么版权问题，但还是请注意一下，比如事先询问作者，又或者作者在模型描述中明确写明了可行的用途。*

### 2.3 数据集准备

准备的训练数据，格式尽量为wav，不同的说话人建立不同的文件夹，每条语音控制在30秒内，确保语音不要有噪音或尽量降低噪音，一个文件夹内语音必须是一个人说的，**语音的质量比起数量来说更重要**。

将准备好的语音文件以以下文件结构将数据集放入`dataset_raw`目录中

```?
dataset_raw
├───speaker0
│   ├───xxx1-xxx1.wav
│   ├───...
│   └───Lxx-0xx8.wav
└───speaker1
    ├───xx2-0xxx2.wav
    ├───...
    └───xxx7-xxx007.wav
```

完成后需要在`.\dataset_raw`文件夹内新建并编辑`config.json`，代码如下：

```json
"n_speakers": 10    //修改数字为说话人的人数
"spk":{
    "speaker0": 0,  //修改speaker0为说话人的名字，需要和文件夹名字一样
    "speaker1": 1,  //值为编号
    "taoxisama": 2,
    "YajuuSenpai": 3,
    "xxx": 4,
    //以此类推
}
```

### 2.4数据预处理

*以下步骤若出现报错请多次尝试，若一直报错则是环境依赖问题，可以根据报错内容使用搜索引擎查找解决方案或重新安装对应的库。*

1. 重采样至 44100hz

```shell
python resample.py
```

2. 自动划分训练集 验证集 测试集 以及自动生成配置文件

```shell
python preprocess_flist_config.py
```

3. 生成hubert与f0

```shell
python preprocess_hubert_f0.py
```

执行完以上步骤后 dataset 目录便是预处理完成的数据，可以删除`dataset_raw`文件夹了。

### 2.5训练

可根据需求修改`.\config\config.json` 配置文件中`train`下的相关值以调整训练参数。

```json
"train": {
    "log_interval": 200,
    "eval_interval": 1000,
    "seed": 1234,
    "epochs": 10000, //训练次数
    "learning_rate": 0.0001, //训练率，不建议过高，建议0.0001
```

运行以下命令开始训练

```shell
python train.py -c configs/config.json -m 44k
```

训练时会自动清除老的模型，只保留最新3个模型，如果想防止过拟合可以修改配置文件`keep_ckpts 0`为永不清除。

一般情况下训练6000次就能得到一个不错的声音模型（有时可能训练出的模型会导致输出僵尸音，需要重新训练，*可能尸变是概率事件*）。

### 2.6推理

##### 2.6.1 分离人声

- 将需要推理的原歌曲文件处理为干声（人声），推荐使用[UVR5](https://github.com/Anjok07/ultimatevocalremovergui)分离人声。

- 将人声分割为若干个不长于40秒的片段，放入`.\raw`文件夹中。可使用FFmpeg进行分割。

---

*以下是使用FFmpeg进行分割的示例：*

```shell
ffmpeg -i "simpal_raw.mp3" -f segment -segment_time 30 -map 0 -c copy %03d.wav
```

该命令将会把原始文件分割为多个最长为30秒的文件。

---

*以下是UVR5分离人声推荐步骤：（如果没有下列步骤的模型请在UVR5中下载）*

- **第一步处理：**
1. Process Method: Demucs

2. Stem: Vocals

3. Demucs Model: v3|UVR-Model-1

4. 勾选 GPU Conversion (使用GPU)

5. Start
- **第二部处理（进一步消除混响与和声）：**
1. 选择第一步处理后的纯人声文件

2. Process Method: VR Architecture

3. Windows Size: 320

4. Aggression: 10

5. VR Model: 5_HP_Karaoke_UVR

6. 勾选 GPU Conversion (使用GPU)

7. 勾选 Vocals Only

8. Start

该方法能尽可能提取出歌曲中较为干净的人声

---

##### 2.6.2 开始推理

- 使用 `inference_main.py`

4.0版本的推理增加了命令行支持

建议直接修改项目目录下的`inference_main.py` 文件。（见文件中注释对应修改，不建议修改`#不用动的部分`的内容）

训练模型建议选择最后生成的`G_xxx.pth`

```shell
python inference_main.py
```

你也可以使用命令行参数进行推理

```shell
# 例
python inference_main.py -m "logs/44k/G_30400.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "nen"
```

必填项部分

- -m, --model_path：模型路径。
- -c, --config_path：配置文件路径。
- -n, --clean_names：wav 文件名列表，放在 raw 文件夹下。
- -t, --trans：音高调整，支持正负（半音）。
- -s, --spk_list：合成目标说话人名称。

可选项部分：见下一节

- -a, --auto_predict_f0：语音转换自动预测音高，转换歌声时不要打开这个会严重跑调。
- -cm, --cluster_model_path：聚类模型路径，如果没有训练聚类则随便填。
- -cr, --cluster_infer_ratio：聚类方案占比，范围 0-1，若没有训练聚类模型则填 0 即可。

**推理生成所得音频文件在`.\results`目录中。**

### 2.7（可选）增强

*以下摘自官方文档*

如果前面的效果已经满意，或者没看明白下面在讲啥，那后面的内容都可以忽略，不影响模型使用(这些可选项影响比较小，可能在某些特定数据上有点效果，但大部分情况似乎都感知不太明显)

#### 自动f0预测

4.0模型训练过程会训练一个f0预测器，对于语音转换可以开启自动音高预测，如果效果不好也可以使用手动的，但转换歌声时请不要启用此功能！！！会严重跑调！！

- 在`inference_main`中设置`auto_predict_f0`为`true`即可

#### 聚类音色泄漏控制

介绍：聚类方案可以减小音色泄漏，使得模型训练出来更像目标的音色（但其实不是特别明显），但是单纯的聚类方案会降低模型的咬字（会口齿不清）（这个很明显），本模型采用了融合的方式， 可以线性控制聚类方案与非聚类方案的占比，也就是可以手动在"像目标音色" 和 "咬字清晰" 之间调整比例，找到合适的折中点。

使用聚类前面的已有步骤不用进行任何的变动，只需要额外训练一个聚类模型，虽然效果比较有限，但训练成本也比较低

- 训练过程：
  - 使用cpu性能较好的机器训练，据我的经验在腾讯云6核cpu训练每个speaker需要约4分钟即可完成训练
  - 执行`python cluster/train_cluster.py` ，模型的输出会在 `logs/44k/kmeans_10000.pt`
- 推理过程：
  - `inference_main`中指定`cluster_model_path`
  - `inference_main`中指定`cluster_infer_ratio`，0为完全不使用聚类，1为只使用聚类，通常设置0.5即可

## 3.后期处理

将生成的干音和歌曲伴奏（也可以通过UVR5提取）导入音频处理软件（如Au等）进行混音和母带处理，最终得到成品。

***祝你武运昌隆！***
