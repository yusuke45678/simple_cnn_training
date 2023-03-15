# A simple CNN training code

CNNを使って学習する単純な練習用コードです．

## requirements

- pytorch
  - torchvision
  - pytorch lightning
- comet
- tqdm

## usage example

```bash
python3 main.py -w 24 -b 8 -e 5 -d ImageFolder -r /mnt/NAS-TVS872XT/dataset-lab/Tiny-ImageNet/  --use_dp
python3 main_pl.py -w 24 -b 8 -e 5 -d ImageFolder -r /mnt/NAS-TVS872XT/dataset-lab/Tiny-ImageNet/ --gpu_strategy ddp --gpus 3
```

`_pl`がついたファイルはPytorch lightningのコードを使用（ddp対応）．

### option

詳しくは`args.py`を参照．主なオプションは以下の通り．

- `-r`：データセットのrootフォルダ
- `-b`：バッチサイズ
- `-w`：データローダーのワーカー数
- `-e`：エポック数
- `-d`：データセット
  - `CIFAR10`：[torchvisionのCIFAR10](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html)
  - `ImageFolder`：`-r`で指定したフォルダ以下に`train/`と`val/`のディレクトリがあり，それ以下はカテゴリ名のサブディレクトリに分かれて保存されている画像データセット（[torchvisionのImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html)）
- `--gpu_strategy`：dp (Data Parallel), ddp (Distributed Data Parallel), single GPU
- `--gpus`: dpとddpで使用するGPU数

または

```bash
python3 main.py -h
python3 main_pl.py -h
```

を実行．

```text
  -h, --help            show this help message and exit
  -r str, --root str    root of dataset. (default: ./downloaded_data)
  -d {CIFAR10,ImageFolder}, --dataset_name {CIFAR10,ImageFolder}
                        name of dataset. (default: CIFAR10)
  -td str, --train_dir str
                        subdier name of training dataset. (default: train)
  -vd str, --val_dir str
                        subdier name of validation dataset. (default: val)
  --torch_home str      TORCH_HOME environment variable where pre-trained model weights are stored. (default: ./pretrained_models)
  -m {resnet18,resnet50}, --model {resnet18,resnet50}
                        CNN model. (default: resnet18)
  --use_pretrained      use pretrained model weights (default) (default: True)
  --scratch             do not use pretrained model weights, instead train from scratch (not default) (default: True)
  -b int, --batch_size int
                        batch size. (default: 8)
  -w int, --num_workers int
                        number of workers. (default: 2)
  -e int, --num_epochs int
                        number of epochs. (default: 25)
  -vi int, --val_interval_epochs int
                        validation interval in epochs. (default: 2)
  -li int, --log_interval_steps int
                        logging interval in steps. (default: 10)
  --optimizer {SGD,Adam}
                        optimizer. (default: SGD)
  --grad_accum int      steps to accumlate gradients. (default: 1)
  -lr float             learning rate. (default: 0.0001)
  --momentum float      momentum of SGD. (default: 0.9)
  --betas float [float ...]
                        betas of Adam. (default: [0.9, 0.999])
  --use_scheduler       use scheduler (not default) (default: False)
  --no_scheduler        do not use scheduler (default) (default: False)
  --use_dp              use multi GPUs with data parallel (default: None)
  --single_gpu          use single GPU (default: None)
  --gpu_strategy {None,dp,ddp}
                        GPU training strategy. None: single GPU. dp: Data Parallel (default). ddp: Distributed Data Parallel. (default: dp)
  --gpus int            how many GPUs are used for dp and ddp. (default: 2)
  --comet_log_dir str   dir to comet log files. (default: ./comet_logs/)
  --tf_log_dir str      dir to TensorBoard log files. (default: ./tf_logs/)
  --save_checkpoint_dir str
                        dir to save checkpoint files. (default: ./log)
  --resume_from_checkpoint str
                        path to the checkpoint file to resume from. (default: None)
  --disable_comet, --no_comet
                        do not use comet.ml (default: use comet) (default: False)
```

## Cometの設定

このディレクトリの`./.comet.config`と，ホームの`~/.comet.config`を利用する．詳しくは[cometのドキュメント](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/advanced/configuration/)を参照．コード中にはAPIキーなどは書かないこと（[logger.py](./logger.py)参照）．

- `~/.comet.config`：すべてに共通する設定を書く．
  - cometのAPIキー，デフォルトのcomet workspaceを設定．
  - `hide_api_key`はTrueにすること（しないとログにAPIキーが残ってしまう）

```ini
[comet]
api_key=XXXXXXXXXXXXXXXXXXXX
workspace=tttamaki

[comet_logging]
hide_api_key=True
```

- このディレクトリの`./.comet.config`：このディレクトリで使用する設定を書く．
  - comet project nameを設定．
  - （ここで設定する内容はホームの`~/.comet.config`よりも優先されて，上書きされる）

```ini
[comet]
project_name=simple_cnn_20230309

[comet_logging]
display_summary_level=0
file=comet_logs/comet_{project}_{datetime}.log

[comet_auto_log]
env_details=True
env_gpu=True
env_host=True
env_cpu=True
cli_arguments=True
```

- コード
  - `Experiment`オブジェクトにcomet experiment nameを設定．必要ならtagを設定する．
  - コード中にはAPIキーなどは書かない．
  - （コード中で設定する内容はこのディレクトリの`./.comet.config`よりも優先されて，上書きされる）

```python
    experiment = Experiment()  # ここでは何も設定しない

    exp_name = datetime.now().strftime('%Y-%m-%d_%H:%M:%S:%f')  # これは日時をexperiment nameに設定する例．
    experiment.set_name(exp_name)
    experiment.add_tag(args.model)  # これはモデル名をタグに設定する例．
```
