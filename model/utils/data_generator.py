import time
import os
import numpy as np
from scipy.misc import imread


from .preprocess import greyscale, get_form_prepro
from .data import minibatches, pad_batch_images, pad_batch_formulas, render
from .images import build_images
from .general import init_dir


class DataGeneratorFile(object):
    """Simple Generator of tuples (img_path, formula_id)"""

    def __init__(self, filename, single=True):
        """Inits Data Generator File

        Iterator that returns
            tuple (img_path, formula_id) if single
            list of (tuples path), else

        Args:
            filename: (string of path to file) where we have
                multiple instances per example
                    aqmsldfj.png 1 qmsdljfs.png 2
                    qsfamsqsdf.png 4 qaezqd.png 5
            single: if multiple instance, false

        """
        self._filename = filename
        self._single = single

    def __iter__(self):
        """
        Return type tuple or list of tuples if more than one instance
        per example
        """
        with open(self._filename) as f:
            for line in f:
                line = line.strip().split(' ')
                instances = []
                for i in range(len(line)/2):
                    instances.append((line[2*i], line[2*i+1]))

                if self._single:
                    yield instances[0]
                else:
                    yield instances



class DataGenerator(object):
    """Data Generator of tuple (image, formula)"""

    def __init__(self, path_formulas, dir_images, path_matching, bucket=False,
                form_prepro=lambda s: s.strip().split(' '), iter_mode="data",
                img_prepro=lambda x: x, max_iter=None, max_len=None,
                bucket_size=20, single=True):
        """Initializes the DataGenerator

        Args:
            path_formulas: (string) file of formulas. There are 2 possible
                settings:
                    - single (one instance per example): one line, one formula
                    - a few instances per example (for instance truth and some
                        proposals): empty line = new example, new line = new
                        instance
            dir_images: (string) dir of images, contains jpg files.
            path_matching: (string) file of name_of_img, id_formula
            img_prepro: (lambda function) takes an array -> an array. Default,
                identity
            form_prepro: (lambda function) takes a string -> array of int32.
                Default, identity.
            max_iter: (int) maximum numbers of elements in the dataset
            max_len: (int) maximum length of a formula in the dataset
                if longer, not yielded.
            iter_mode: (string) "data", "full" to set the type returned by the
                generator
            bucket: (bool) decides if bucket the data by size of image
            bucket_size: (int)
            single: (bool) if each example contains multiple (images, formulas)

        """
        self._path_formulas  = path_formulas
        self._dir_images     = dir_images
        self._path_matching  = path_matching
        self._img_prepro     = img_prepro
        self._form_prepro    = form_prepro
        self._max_iter       = max_iter
        self._max_len        = max_len
        self._iter_mode      = iter_mode
        self._bucket         = bucket
        self._bucket_size    = bucket_size
        self._single         = single

        self._length         = None
        self._formulas       = self._load_formulas(path_formulas)

        self._set_data_generator()


    def _set_data_generator(self):
        """Sets iterable or generator of tuples (img_path, id of formula)"""
        self._data_generator = DataGeneratorFile(self._path_matching,
                self._single)

        if self._bucket:
            self._data_generator = self.bucket(self._bucket_size)


    def bucket(self, bucket_size):
        """Iterates over the listing and creates buckets of same shape images.

        Args:
            bucket_size: (int) size of the bucket

        Returns:
            bucketed_dataset: [(img_path1, id1), ...]

        """
        print("Bucketing the dataset...")
        bucketed_dataset = []
        old_mode = self._iter_mode # store the old iteration mode
        self._iter_mode = "full"

        # iterate over the dataset in "full" mode and create buckets
        data_buckets = dict() # buffer for buckets
        for idx, (img, formula, img_path, formula_id) in enumerate(self):
            s = img.shape
            if s not in data_buckets:
                data_buckets[s] = []
            # if bucket is full, write it and empty it
            if len(data_buckets[s]) == bucket_size:
                for (img_path, formula_id) in data_buckets[s]:
                    bucketed_dataset += [(img_path, formula_id)]
                data_buckets[s] = []

            data_buckets[s] += [(img_path, formula_id)]

        # write the rest of the buffer
        for k, v in data_buckets.iteritems():
            for (img_path, formula_id) in v:
                bucketed_dataset += [(img_path, formula_id)]


        self._iter_mode = old_mode
        self._length    = idx

        print("- done.")
        return bucketed_dataset


    def _load_formulas(self, filename):
        """Loads txt file with formulas in a dict

        In the multiple instance per example setting, between 2 example, there
        is an id missing, that's how you know that the formulas belong to an
        other example.

        Args:
            filename: (string) path of formulas. There are 2 possible
                settings:
                    - single (one instance per example): one line, one formula
                    - a few instances per example (for instance truth and some
                        proposals): empty line = new example, new line = new
                        instance

        Returns:
            dict: dict[idx] = one formula

        """
        formulas = dict()
        with open(filename) as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if len(line) != 0:
                    formulas[idx] = line

        return formulas


    def _process_instance(self, example):
        """From path and formula id, returns actual data

        Applies preprocessing to both image and formula

        Args:
            example: tuple (img_path, formula_ids)
                img_path: (string) path to image
                formula_id: (int) id of the formula

        Returns:
            img: depending on _img_prepro
            formula: depending on _form_prepro

        """
        img_path, formula_id = example

        img = imread(self._dir_images + "/" + img_path)
        img = self._img_prepro(img)

        formula = self._form_prepro(self._formulas[int(formula_id)])

        if self._iter_mode == "data":
            inst = (img, formula)
        elif self._iter_mode == "full":
            inst = (img, formula, img_path, formula_id)

        # filter on the formula length
        if self._max_len is not None and len(formula) > self._max_len:
            skip = True
        else:
            skip = False

        return inst, skip


    def __iter__(self):
        """Iterator over Dataset

        Yields:
            tuple (img, formula) if _single
                img: array
                formula: one formula
            or list of those tuples else

        """
        n_iter = 0
        for example in self._data_generator:
            if self._max_iter is not None and n_iter >= self._max_iter:
                break
            # just one training instance per line
            if type(example) is tuple:
                result, skip = self._process_instance(example)
                if skip: continue

            # multiple instances of the same example per line
            elif type(example) is list:
                result, skip = zip(*[self._process_instance(inst) for inst
                        in example])
                if skip[0]: continue # filter on first example only

            n_iter += 1
            yield result


    def __len__(self):
        if self._length is None:
            print("First call to len(dataset) - may take a while.")
            counter = 0
            for _ in self:
                counter += 1
            self._length = counter
            print("- done.")

        return self._length


    def build(self, single=True, quality=100, density=200, down_ratio=2,
        buckets=None, n_threads=4):
        """Generates images from the formulas and writes the correspondance
        in the matching file.

        Args:
            single: if there is one instance per example (one formula only)
            quality: parameter for magick
            density: parameter for magick
            down_ratio: (int) downsampling ratio
            buckets: list of tuples (list of sizes) to produce similar shape images

        """
        # 1. produce images
        init_dir(self._dir_images)
        result = build_images(self._formulas, self._dir_images, quality,
                density, down_ratio, buckets, n_threads)

        # 2. write matching with same convention of naming
        with open(self._path_matching, "w") as f:
            if self._single:
                for (path_img, idx) in result:
                    if path_img is not False: # image was successfully produced
                        f.write("{} {}\n".format(path_img, idx))
            else:
                last_idx, example_match = -1, []
                for idx, formula in self._formulas.items():
                    path_img = str(idx) + ".png"

                    if idx - last_idx > 1 and len(example_match) > 0:
                        f.write(" ".join(example_match) + "\n")
                        example_match = [path_img, idx]
                    else:
                        example_match += [path_img, idx]

                    last_idx = idx
