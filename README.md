# Im2Latex

Seq2Seq model with Attention + Beam Search for Image to LaTeX.

[Blog post](https://guillaumegenthial.github.io/image-to-latex.html)

## Install

Install pdflatex (latex to pdf) and ghostsript + [magick](https://www.imagemagick.org/script/install-source.php
) (pdf to png) on Linux


```
make install-linux
```

(takes a while, installs from source)

On Mac, assuming you already have a LaTeX distribution installed, you should have pdflatex and ghostscript installed, so you just need to install magick. You can try

```
make install-mac
```

## Data

You can download the [prebuilt dataset from Harvard](https://zenodo.org/record/56198#.V2p0KTXT6eA) and use their preprocessing scripts found [here](https://github.com/harvardnlp/im2markup)


## Getting Started

We provide a small dataset just to check the pipeline. If you haven't touched the files, run

```
make run
```

or perform the following steps

1. Build the images from the formulas, write the matching file and extract the vocabulary. __Run only once__
```
python build.py
```

2. Train on this small dataset
```
python train.py
```

3. Evaluate the text metrics
```
python evaluate_txt.py
```

4. Evaluate the image metrics
```
python evaluate_img.py
```

You should observe that the model starts to produce reasonable patterns of LaTeX.

## Config

Edit the config files in configs/ for your needs and change the name of the config files used in `build.py`, `train.py` etc.
