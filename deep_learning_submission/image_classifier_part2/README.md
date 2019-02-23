# Usage

## `train.py`

```console
# python train.py -h
usage: train.py [-h] [--learning_rate LEARNING_RATE]
                [--hidden_units HIDDEN_UNITS] [--epochs EPOCHS] [--gpu]
                [--arch ARCH] [--save_dir SAVE_DIR]
                data_dir

Train Argparser

positional arguments:
  data_dir              path to directory where train validation and test
                        images are stored

optional arguments:
  -h, --help            show this help message and exit
  --learning_rate LEARNING_RATE
                        Learning rate for optimizer
  --hidden_units HIDDEN_UNITS
                        Number of perceptrons in the hidden layer
  --epochs EPOCHS       How many times we need to iterate over the entire
                        training set
  --gpu                 Enable GPU (cuda) for training
  --arch ARCH           Pre-trained model architecture
  --save_dir SAVE_DIR   path to directory where we save our trained model
```

### how to use `train.py`


Provide a `data_dir` path where there is `train` subdirectory that holds trainin images in category dirs.
All other arguments are optional but important

Train a model with `arch` `vgg11` and `hidden_units` has 2048 perceptrons. Run this for 3 `epochs` and set `learning_rate` to 0.001. Use `gpu` to train the model.

```console
# python train.py flowers --arch=vgg11 --learning_rate=0.001 --hidden_units=2048 --epochs=3 --gpu --save_dir=saved_models
Training vgg11 model for 3 epoch(s).
 # of hidden perceptrons: 2048
Learning Rate: 0.001
Training with GPU set to True
Epoch 1/3.. Train loss: 5.967.. Validation loss: 4.498.. Validation accuracy: 0.025
Epoch 1/3.. Train loss: 4.402.. Validation loss: 4.120.. Validation accuracy: 0.098
Epoch 1/3.. Train loss: 4.002.. Validation loss: 3.560.. Validation accuracy: 0.212
Epoch 1/3.. Train loss: 3.596.. Validation loss: 2.928.. Validation accuracy: 0.322
Epoch 1/3.. Train loss: 3.136.. Validation loss: 2.418.. Validation accuracy: 0.417
Epoch 1/3.. Train loss: 2.920.. Validation loss: 2.091.. Validation accuracy: 0.482
Epoch 1/3.. Train loss: 2.509.. Validation loss: 1.796.. Validation accuracy: 0.527
...etc
```



## `predict.py`

```console
# python predict.py -h
usage: predict.py [-h] [--top_k TOP_K] [--category_names CATEGORY_NAMES]
                  [--gpu]
                  input checkpoint

Predict Argparser

positional arguments:
  input                 path to image to predict
  checkpoint            checkpoint name, search path is set to ./saved_models

optional arguments:
  -h, --help            show this help message and exit
  --top_k TOP_K         Top K mostly likely classes
  --category_names CATEGORY_NAMES
                        Mapping of categories to real names
  --gpu                 Enable GPU (cuda) for prediction
```

### how to use (`predict.py`)

Predict the flower category, pick a image from 43th category in test set using trained model (checkpoint) `vgg13`,
a rather poorly trained model, using epoch=1.


```console
#python predict.py flowers/test/43/image_02371.jpg vgg13 --gpu --top_k=1
(0.2741462290287018, '44')
```

Indeed we got wrong category, test image is in category `43` and prediction is `44`

Try to predict the same flower category using a different model `vgg11`. This model trained a bit longer (epoch=3)
and has more prediction power.
Result shows that top prediction is category `43` (correct) and probability assigned to this category is `0.17`.

```console
#python predict.py flowers/test/43/image_02371.jpg vgg11 --gpu --top_k=1
(0.17053773999214172, '43')

```

For the same image get top 3 predictions: category `101` and `96` are in the top three.

```console
# python predict.py flowers/test/43/image_02371.jpg vgg11 --gpu --top_k=3
([0.17053773999214172, 0.12676630914211273, 0.0725887194275856], ['43', '101', '96'])

```

Get actual names of the flowers: Looks like category `43` is *sword lily*

```console
# python predict.py flowers/test/43/image_02371.jpg vgg11 --gpu --top_k=3 --category_names=cat_to_name.json
([0.17053773999214172, 0.12676630914211273, 0.0725887194275856], ['sword lily', 'trumpet creeper', 'camellia'])
```

