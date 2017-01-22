# ultrasound-nerve-segmentation
Kaggle ultrasound nerve segmentation challenge using Keras. 
Read my [blog](https://raghakot.github.io/2016/12/26/Ultrasound-nerve-segmentation-challenge-on-Kaggle.html) for 
details and insights.

#Install (Ubuntu {14,16}, GPU)

cuDNN required.

###Tensorflow backend
- https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md

###Keras
- sudo apt-get install libhdf5-dev
- sudo pip install h5py
- sudo pip install keras

In ~/.keras/keras.json
```
{
    "image_dim_ordering": "th",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

###Python deps
 - sudo apt-get install python-opencv
 - sudo apt-get install python-sklearn

#Prepare
Download the data from https://www.kaggle.com/c/ultrasound-nerve-segmentation/data
and place it in input/train and input/test folders respectively.

Run
```
python data.py
```
to generate data within input folder. This is a one time only operation,

#Training

```
python train.py
```
Results will be generated in "results/" folder. results/net.hdf5 - best model

#Submission
```
python submission.py
```
will generate submission with run length encoding that can directly be uploaded to kaggle.

#Model

I used U-net like architecture (http://arxiv.org/abs/1505.04597) with a few tweaks.
 - Main idea was to use two training heads, one optimizing bce for nerve presence and other optimizing dice for segmentation.
During test time simply zero out masks that have probability < 0.5. This was necessary because large number of samples
contained no masks, and bce/dice score alone would simply be optimized by outputting all zeros for masks.
 - Network contains ~8.25 million parameters. Single epoch took 4 minutes on a Titan X with 12 GB memory.
 - Reduced learning rate by factor of 0.25 when stagnation occurred within last 4 epochs.
 - Logs are written to 'logs/' folder and monitored via tensorboard. Examined histograms to detect saturation. Note that
 you need to use fixed set vs generator to get histograms, as of keras 1.1.1 due to a known issue.
 - Weight regularization prevented convergence (perhaps smaller lambda needed to be used).
 Used dropout instead to prevent weight saturation (which tended to occur without it)
 - he_normal weight initialization.
 - conv with 2 X 2 stride instead of max pooling to downsample, in light of recent results with VAE and GANs.
 - ELU activation, batchnorm everywhere.
 - Used 1 X 1 conv instead of dense layers in the spirit of paper - "Striving for simplicity - The all conv net".

Augmentation:
 - Parallel aug generation on CPU.
 - random rotation (+/- 5 deg)
 - random translations (+/- 10 px)
 - elastic deformation didn't help much.
 - Larger rotations/translations prevented learning.

Validation:
- 10% of the examples, stratified split by mask/no-mask

# Visual inspection:
- utils.examine_generator() can be used to visually inspect augmented samples.
- utils.inspect_set() can be used to examine test time predictions on train/val set.
- I am in the process of generalizing layer visualization code. Otherwise, I used various gradient ascent style
visualizations to sanity check if the network is learning the right thing.

#Credits
Borrowed starter code from https://github.com/jocicmarko/ultrasound-nerve-segmentation/, particularly data prep and
submission portion.