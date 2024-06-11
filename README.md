# GazeTR

ICPR2022で採択された"**Gaze Estimation using Transformer**"の研究において、GazeTR-Hybridのコードを提供しています。

データ処理には、<a href="http://phi-ai.org/GazeHub/" target="_blank">*GazeHub*</a>で提供されているコードを使用することを推奨します。処理済みデータセットを使用して、直接メソッドのコードを実行できます。

*注意: 一部のユーザーは、ブラウザでコードが開けない問題が発生することがあります。その場合、直接コードをダウンロードし、ファイル名を `xx.py` に変更してください。実際には、ダウンロードステップは自動的に処理されますが、ブラウザによってこの問題が発生することがあります。*

<div align=center> <img src="src/overview.png"> </div>

## Requirements
このプロジェクトは pytorch1.7.0 を使用して構築します。

`warmup` は <a href="https://github.com/ildoonet/pytorch-gradual-warmup-lr" target="_blank">こちら</a>を参考に使用します。



## 使用方法(Usage)
### チュートリアル
```
git clone "https://github.com/Morata91/MyGazeTR.git"
cd MyGazeTR
```

```
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```
インストールされるもの
・PyTorch
・TorchVision
・Numpy
・OpenCV
・YAML
・easydict
・warmup-scheduler

### 直接コードを使用する場合

コードを実行するために、以下の3つのステップを行う必要があります。

1.	提供されたデータ処理コードを使用してデータを準備します。
2.	`config/train/config_xx.yaml` と `config/test/config_xx.yaml` を修正します。
3.	コマンドを実行します。

leave-one-person-out評価を行うには、以下のコマンドを実行します。
*学習*
```
python trainer/leave.py -s config/train/config_xx.yaml -p 0
```
*テスト*
```
python tester/leave.py -s config/train/config_xx.yaml -t config/test/config_xx.yaml -p 0
```

このコマンドは`0`番目の人以外のデータで学習し、`0`番目の人のデータでテストします。`-p` のパラメータを変更して繰り返す必要があります。

トレーニングテスト評価を行うには、以下のコマンドを実行します。
To perform training-test evaluation, you can run
*学習*
```
python trainer/total.py -s config/train/config_xx.yaml    
```

*テスト*
```
python tester/total.py -s config/train/config_xx.yaml -t config/test/config_xx.yaml
```

### 独自のプロジェクトを構築する場合
model.py 内のモデルを自身のプロジェクトにインポートすることができます。
You can import the model in `model.py` for your own project.

以下に例を示します。なお、model.py の line 114 は .cuda() を使用しています。CPUでモデルを実行する場合はこれを削除する必要があります。
We give an example. Note that, the `line 114` in `model.py` uses `.cuda()`. You should remove it if you run the model in CPU.
```
from model import Model
GazeTR = Model()

img = torch.ones(10, 3, 224 ,224).cuda()
img = {'face': img}
label = torch.ones(10, 2).cuda()

# for training
loss = GazeTR(img, label)

# for test
gaze = GazeTR(img)
```

## 事前学習済モデル(Pre-trained model)
Google Drive  または  Baidu Cloud Disk  からコード 1234 を使用してダウンロードできます。
You can download from <a href="https://drive.google.com/file/d/1WEiKZ8Ga0foNmxM7xFabI4D5ajThWAWj/view?usp=sharing" target="_blank"> google drive </a> or <a href="https://pan.baidu.com/s/1GEbjbNgXvVkisVWGtTJm7g" target="_blank"> baidu cloud disk </a> with code `1234`. 

これは、ETH-XGazeデータセットで50エポックと512バッチサイズで事前学習されたモデルです。
This is the pre-trained model in ETH-XGaze dataset with 50 epochs and 512 batch sizes. 

## パフォーマンス(Performance)
![ComparisonA](src/ComparisonA.png)

![ComparisonB](src/ComparisonB.png)

## 引用(Citation)
```
@InProceedings{cheng2022gazetr,
  title={Gaze Estimation using Transformer},
  author={Yihua Cheng and Feng Lu},
  journal={International Conference on Pattern Recognition (ICPR)},
  year={2022}
}
```

## Links to gaze estimation codes.

- A Coarse-to-fine Adaptive Network for Appearance-based Gaze Estimation, AAAI 2020 (Coming soon)
- [Gaze360: Physically Unconstrained Gaze Estimation in the Wild](https://github.com/yihuacheng/Gaze360), ICCV 2019
- [Appearance-Based Gaze Estimation Using Dilated-Convolutions](https://github.com/yihuacheng/Dilated-Net), ACCV 2019
- [Appearance-Based Gaze Estimation via Evaluation-Guided Asymmetric Regression](https://github.com/yihuacheng/ARE-GazeEstimation), ECCV 2018
- [RT-GENE: Real-Time Eye Gaze Estimation in Natural Environments](https://github.com/yihuacheng/RT-Gene), ECCV 2018
- [MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation](https://github.com/yihuacheng/Gaze-Net), TPAMI 2017
- [It’s written all over your face: Full-face appearance-based gaze estimation](https://github.com/yihuacheng/Full-face), CVPRW 2017
- [Eye Tracking for Everyone](https://github.com/yihuacheng/Itracker), CVPR 2016
- [Appearance-Based Gaze Estimation in the Wild](https://github.com/yihuacheng/Mnist), CVPR 2015

## License
The code is under the license of [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Contact 
Please email any questions or comments to yihua_c@buaa.edu.cn.
