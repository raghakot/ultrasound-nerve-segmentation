# ultrasound-nerve-segmentation
Kaggle ultrasound nerve segmentation challenge using Keras. Detailed insights including visualization
and experimentation will follow shortly on my blog.

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
to generate data within input folder

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
 - Batch norm and ELU
 - 2 heads training: auxiliary branch for scoring nerve presence (in the middle of the network), one branch for segmentation
 - Custom Dice loss coefficient, experimented with both strict (avg of dice per image), and average over batch.
 Smoothing is used to avoid discontinuities when there is no nerve.
 - batch_size = 32
 - Ran on Titan X GPU with 12 GB memory.

Augmentation:
 - flip x,y
 - random rotation (+/- 10 deg)
 - random translations (+/- 20 px)
 - elastic deformation didn't help much.

Validation:
- Stratified split by mash/no-mask
- Split by patient id didn't workout

# Results and technical details
- Network contains ~21.8 million parameters. Single epoch took 4.8 minutes.
- Reduced learning rate by factor of 0.25 when stagnation occurred within last 4 epochs.
- Logs are written to logs/ and monitored via tensorboard. Examined histograms to detect saturation.
- Best single model achieved ~0.7 LB score, which puts you on top 30. Ensembling improves this to ~0.71 ish.


#Credits
Borrowed starter code from https://github.com/jocicmarko/ultrasound-nerve-segmentation/, particularly data prep and
submission portion.