# SAMPN_PyTorch

This is a torn down version of the original SAMPN repository (found [here](https://github.com/tbwxmu/SAMPN)). We also provide an improved model, which we refer to as `QSARPlus` following SAMPN's model titled `QSAR`.

## Training
To train a model:
```bash
usage: train.py [-h] [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                [--warmup-epochs WARMUP_EPOCHS] [--dataset DATASET]

Train a model on a dataset.

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number of epochs to train.
  --batch-size BATCH_SIZE
                        Batch size.
  --warmup-epochs WARMUP_EPOCHS
                        Number of epochs to warmup.
  --dataset DATASET     Dataset type (lipophilicity or solubility).
```

Training results on lipophilicity using QSAR:

![](./screenshots/sampn_lipophilicity_train_loss.png)

![](./screenshots/sampn_lipophilicity_val_loss.png)

Training results on aqueous solubility using QSAR:

![](./screenshots/sampn_solubility_train_loss.png)

![](./screenshots/sampn_solubility_val_loss.png)

## Testing
To test a model:
```bash
usage: test.py [-h] [--batch-size BATCH_SIZE] [--dataset DATASET] [--model MODEL]
               [--weights WEIGHTS] [--iterations ITERATIONS]

Test a model on a dataset.

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        Batch size.
  --dataset DATASET     Dataset type (lipophilicity or solubility).
  --model MODEL         Model type (QSAR or QSARPlus).
  --weights WEIGHTS     Path to the model weights.
  --iterations ITERATIONS
                        Number of iterations to run.
```

Our test results on lipophilicity:

```bash
$ python3 test.py --dataset lipophilicity --model QSAR --weights ./checkpoint/lipophilicity/QSAR_state_dict.pt
rmse       0.2979
mae        0.2188
mse        0.0894
r2         0.9211
pearson    0.9636

$ python3 test.py --dataset lipophilicity --model QSARPlus --weights ./checkpoint/lipophilicity/QSARPlus_state_dict.pt
rmse       0.5763
mae        0.5763
mse        0.5524
```

Our test results on aqueous solubility:
```bash
$ python3 test.py --dataset solubility --model QSAR --weights ./checkpoint/solubility/QSAR_state_dict.pt
rmse       0.7650
mae        0.6428
mse        0.5932
r2         0.2540
pearson    0.7287

$ python3 test.py --dataset solubility --model QSARPlus --weights ./checkpoint/solubility/QSARPlus_state_dict.pt
rmse       0.5362
mae        0.5362
mse        0.6376
```