# Data_Driven_Color_Augmentation

Implementation of H\&E-adversarial network: a convolutional neural network to learn stain-invariant features through Hematoxylin & Eosin regression.

## Reference
If you find this repository useful in your research, please cite:

[1] Marini N., Otálora S., Wodzinski M., Tomassini S. Dragoni A.F., Marchand-Maillet S., Dominguez P., Duran-Lopez L., Vatrano S., Müller H. & Atzori M., Data-driven color augmentation for H&E stained images in computational pathology.

Paper link: https://www.sciencedirect.com/science/article/pii/S2153353922007830

## Requirements
Python==3.6.9, albumentations==0.1.8, numpy==1.17.3, opencv==4.2.0, pandas==0.25.2, pillow==6.1.0, torchvision==0.8.1, pytorch==1.7.0

## CSV Input Files:
CSV files are used as input for the scripts. For each partition (train, validation, test), the csv file has path_to_image, class_label as columns.
For prostate experiments, the class_label can be: 
0: benign
1: Gleason pattern 3
2: Gleason pattern 4
3: Gleason pattern 5

For colon experiments, the class_label can be:
0: cancer
1: dysplasia
2: normal glands

## Augmentation
Methods to perform data drive color augmentation (Augmentation.py):
- new_color_augmentation (HSC color augmentation):
  * patch_np: numpy array for the input patch (224x224)
  * kdtree: the database where acceptable color variations are stored
  * alpha: neighbors
  * beta: radius
  * shift_value: perturbation to apply to Hue, Saturation, Contrast 
- new_stain_augmentation (perturbation of H&E channels):
  * patch_np: numpy array for the input patch (224x224)
  * kdtree: the database where acceptable color variations are stored
  * alpha: neighbors
  * beta: radius
  * sigma1: range (-sigma1, sigma1) to generate random value to multiply to H&E components
  * sigma2: range (-sigma1, sigma1) to generate random value to add to H&E components

## Database
Database including color variations will be uploaded soon Zenodo.

Method to extend database with new histopathology patches (extend_kd_tree_offline)
  * -i: input pickle file (database to extend)
  * -o: output pickle file
  * -d: csv including patches to extend database

## Training
Scripts to train the CNN at path-level, in a fully-supervised fashion.
Some parameters must be manually changed, such as the number of classes (output of the network).

- Training_new_augmentation.py -n -b -c -e -f -i -o -a -d. The script is used to train the CNN without any augmentation (no_augment), with colour augmentation (augment).
  * -n: number of the experiment for the training
  * -b: batch size (32)
  * -c: CNN backbone to use (densenet121)
  * -e: number of epochs (10)
  * -t: task of the network (no_augment, augment, normalizer)
  * -f: if True an embedding layer with 128 nodes is inserted before the output layer
  * -i: path of the folder where the input csvs for training (train.csv), validation (valid.csv) and testing (test.csv) are stored
  * -o: path of the folder where to store the CNN’s weights.
  * -a: new augmentation to use: color (HSC color augmentation), stain (H&E stain augmentation), he (H&E-adversarial CNN + HSC color augmentation)
  * -x: extend (False): add color variations training data to color variation dataset
  * -d: database: database including color variations


## Acknoledgements
This project has received funding from the EuropeanUnion’s Horizon 2020 research and innovation programme under grant agree-ment No. 825292 [ExaMode](http://www.examode.eu). Infrastructure fromthe SURFsara HPC center was used to train the CNN models in parallel. 
