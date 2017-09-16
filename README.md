# Img2Latex

## Install

Install ghostsript and magick (from source depending on your linux distribution) and pdflatex for evaluation

https://www.imagemagick.org/script/install-source.php


```
sudo pip install -r requirements.txt
sudo apt-get install texlive-latex-base
sudo apt-get install texlive-latex-extra

sudo apt-get install ghostscript
sudo apt-get install libgs-dev

wget http://www.imagemagick.org/download/ImageMagick.tar.gz
tar -xvf ImageMagick.tar.gz
cd ImageMagick-7.*
./configure --with-gslib=yes 
make
sudo make install
sudo ldconfig /usr/local/lib
```


## Data and Preprocessing

We use Harvard preprocessing scripts that can be found at http://lstm.seas.harvard.edu/latex/

First, crop + downsampling of images + group by similar shape


```
python scripts/preprocessing/preprocess_images.py --input-dir data/formula_images --output-dir data/images_processed
```

Second, parse formulas with KaTeX parser

```
python scripts/preprocessing/preprocess_formulas.py --mode normalize --input-file data/im2latex_formulas.lst --output-file data/norm.formulas.lst
```

Third, filter formulas

```
python scripts/preprocessing/preprocess_filter.py --filter --image-dir data/images_processed --label-path data/norm.formulas.lst --data-path data/im2latex_train.lst  --output-path data/train_filter.lst
python scripts/preprocessing/preprocess_filter.py --filter --image-dir data/images_processed --label-path data/norm.formulas.lst --data-path data/im2latex_validate.lst  --output-path data/val_filter.lst
python scripts/preprocessing/preprocess_filter.py --filter --image-dir data/images_processed --label-path data/norm.formulas.lst --data-path data/im2latex_test.lst  --output-path data/test_filter.lst
```


## Train

Edit the config file in configs/

```
python main.py
```