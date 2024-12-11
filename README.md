# From Chaos to Clarity: Time Series Anomaly Detection in Astronomical Observations(ICDE 2024)

This is PyTorch implementation of AERO in the following paper:

"From Chaos to Clarity: Time Series Anomaly Detection in Astronomical Observations"

```
@INPROCEEDINGS{10598045,
  author={Hao, Xinli and Chen, Yile and Yang, Chen and Du, Zhihui and Ma, Chaohong and Wu, Chao and Meng, Xiaofeng},
  booktitle={2024 IEEE 40th International Conference on Data Engineering (ICDE)},
  title={From Chaos to Clarity: Time Series Anomaly Detection in Astronomical Observations},
  year={2024},
  volume={},
  number={},
  pages={570-583},
  keywords={Learning systems;Noise;Time series analysis;Stars;Transformers;Data engineering;Graph neural networks;Time series;Anomaly detection;AI for science},
  doi={10.1109/ICDE60146.2024.00050}}


```



## Data Preparation

### Notices:

以下前處理尚未完成，需要後續更正才可使用，目前只有做到 label 的部分

1. 將下載好的 dataset 放置 /dataset/自己命名(your_dataset 舉例)

- [dataset 連結](https://huggingface.co/datasets/helenqu/astro-classification-redshifts-augmented?row=3)
  > 出處 : helenqu/astro-classification-redshifts-augmented

```
dataset
 |-your_dataset
 | |-0000.csv
```

2. 對 dataset 作前處理

```
python3 src/dataset_preprocess.py --pre_dataset_name 0000.csv --pre_dataset_folder your_dataset
```

3. 需要將 dataset/自己命名(your_dataset 舉例)/output.csv 中的資料進行處理(尚未完成)

- 需要取出 幾顆固定的星星 出現在 N 個連續的 timeslot 的資料
- 再將取出的資料分成 test / train 轉成 .txt (可以參考如下)

```
# put your dataset under processed/ directory with the same structure shown in the data/msl/

Dataset_txt
 |-AstrosetMiddle
 | |-AstrosetMiddle_train.txt    # training data
 | |-AstrosetMiddle_test.txt     # test data
 | |-AstrosetMiddle_interpretation_label.txt    # True anomaly label
 |-your_dataset
 | |-XX_train.txt
 | |-XX_test.txt
 | |-XX_interpretation_label.txt
 | ...
```
## Environment settings
原本的環境設定較舊，故改成下方設定，
(requirements.txt)
matplotlib==3.3.4
numpy==1.19.5
pandas==1.1.5
tqdm==4.64.0
scipy==1.5.4
scikit-learn==0.24.2
torch==1.13.0

## Requirements

Dependency can be installed using the following command:

```
pip install -r requirements.txt
```

## Model Training
使用論文提供的資料庫去作訓練，共有合成集與真實集各三個，共六個。

-Parameter description(參數說明)
 1.–dataset_name：指定數據集的名稱
 2.–retrain：啟用重新訓練模式
 3.–freeze_patienc：設置凍結模型層（freeze layers）的耐心次數
 4.–freeze_delta：設置凍結模型層的性能提升閾值
 5.–stop_patience：設置訓練的早停耐心次數
 6.–stop_delta：設置早停的性能提升閾值。
 參數整體邏輯

	1.	數據集選擇：--dataset_name 指定使用的數據集（如 SyntheticMiddle）。
	2.	重新訓練：--retrain 強制重新訓練模型，忽略現有的權重檔案。
	3.	凍結策略：通過 --freeze_patience 和 --freeze_delta 動態凍結部分模型層，提升訓練效率。
	4.	早停策略：通過 --stop_patience 和 --stop_delta 動態停止訓練，避免過度訓練和資源浪費。

整體流程範例

在這段指令中，訓練將針對 SyntheticMiddle 數據集執行，並具備以下行為：

	1.	如果模型性能在 5 次迭代內提升幅度低於 0.01，部分層的參數將凍結。
	2.	如果性能在 5 次迭代內無提升（提升幅度小於 0.01），訓練將提前停止。
	3.	模型不使用之前的訓練結果，而是重新開始訓練。

Command operation指令操作：
- SyntheticMiddle

```
python3 main.py  --dataset_name SyntheticMiddle  --retrain --freeze_patience 5 --freeze_delta 0.01 --stop_patience 5 --stop_delta 0.01
```

python main.py --dataset_name SyntheticMiddle --retrain --freeze_patience 5 --freeze_delta 0.01 --stop_patience 5 --stop_delta 0.01

- SyntheticHigh

```
python3 main.py  --dataset_name SyntheticHigh  --retrain --freeze_patience 5 --freeze_delta 0.01 --stop_patience 5 --stop_delta 0.005
```

- SyntheticLow

```
python3 main.py  --dataset_name SyntheticLow  --retrain --freeze_patience 5 --freeze_delta 0.01 --stop_patience 5 --stop_delta 0.005
```

- AstrosetMiddle

```
python3 main.py  --dataset_name AstrosetMiddle  --retrain --freeze_patience 5 --freeze_delta 0.01 --stop_patience 5 --stop_delta 0.005
```

- AstrosetHigh

```
python3 main.py  --dataset_name AstrosetHigh  --retrain --freeze_patience 5 --freeze_delta 0.01 --stop_patience 5 --stop_delta 0.005
```

- AstrosetLow

```
python3 main.py  --dataset_name AstrosetLow  --retrain --freeze_patience 5 --freeze_delta 0.005 --stop_patience 5 --stop_delta 0.001
```

# data-mining
