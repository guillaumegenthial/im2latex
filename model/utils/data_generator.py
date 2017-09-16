import time
import os
import numpy as np
from scipy.misc import imread

from preprocess import greyscale, get_form_prepro
from data_utils import minibatches, pad_batch_images, \
    load_vocab, pad_batch_formulas, render
from utils.images import convert_to_png


class DataGenerator(object):

    def __init__(self, filename, single=True):
        """Inits Data Generator

        Args:
            filename: (string of path to file) where we have
                multiple instances per example
                    aqmsldfj.png 1 qmsdljfs.png 2
                    qsfamsqsdf.png 4 qaezqd.png 5
            single: if multiple instance, false

        Returns:
            iterator that returns
                tuple img_path, formula_id if n == 2
                tuple list of tuples path, id if n != 2

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


class Dataset(object):
    def __init__(self, path_formulas, dir_images, path_matching,
                img_prepro, form_prepro, max_iter=None, max_len=None,
                iter_mode="data", bucket=True, bucket_size=20, single=True):
        """
        Args:
            path_formulas: (string) file of formulas, one formula per line
            dir_images: (string) dir of images, contains jpg files
            path_matchin: (string) file of name_of_img, id_formula
            img_prepro: (lambda function) takes an array -> an array
            form_prepro: (lambda function) takes a string -> array of int32
            max_iter: (int) maximum numbers of elements in the dataset
            max_len: (int) maximum length of a formula in the dataset
                if longer, not yielded.
        """
        self.path_formulas = path_formulas
        self.dir_images    = dir_images
        self.path_matching = path_matching
        self.img_prepro    = img_prepro
        self.form_prepro   = form_prepro
        self.formulas      = self._load_formulas(path_formulas)
        self.length        = None     # computed when len(Dataset) is called
                                      # for the first time
        self.max_iter      = max_iter # optional
        self.max_len       = max_len  # optional
        self.iter_mode     = iter_mode

        self.data_generator = DataGeneratorFile(self.path_matching, single)

        if bucket:
            self.data_generator = self.bucket(bucket_size)


    def bucket(self, bucket_size):
        """
        Iterates over the listing and creates buckets of same shape
        images.
        Args:
            bucket_size: (int) size of the bucket

        Returns:
            bucketed_dataset: [(img_path1, id1), ...]
        """
        print("Bucketing the dataset...")
        bucketed_dataset = []
        # store the old iteration mode
        old_mode = self.iter_mode
        self.iter_mode = "full"

        # iterate over the dataset and create buckets
        data_buckets = dict()

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

        # write the rest of the data
        for k, v in data_buckets.iteritems():
            for (img_path, formula_id) in v:
                bucketed_dataset += [(img_path, formula_id)]

        print("- done.")
        self.iter_mode = old_mode
        self.length = idx

        return bucketed_dataset


    def _load_formulas(self, filename):
        """
        Args:
            filename: (string) path of formulas, one formula per line
        Returns:
            dict: dict[idx] = one formula
        """
        formulas = dict()
        with open(filename) as f:
            for idx, line in enumerate(f):
                line = line.strip()
                formulas[idx] = line

        return formulas


    def get_max_shape(self):
        """
        Computes max shape of images in the dataset
        Returns:
            max_shape_image: tuple (max_heigh, max_width, max_channels)
                of images in the dataset
            max_length_formula: max length of formulas in the dataset
        """
        max_shape = [0,0,0]
        max_length = 0

        with open(self.path_matching) as f:
            for line in f:
                img = imread(self.dir_images + "/" + img_path)
                img = self.img_prepro(img)
                max_shape[0] = max(max_shape[0], img.shape[0])
                max_shape[1] = max(max_shape[1], img.shape[1])
                max_shape[2] = max(max_shape[2], img.shape[2])
                formula = self.form_prepro(self.formulas[int(formula_id)])
                max_length = max(max_length, len(formula))

        return max_shape, max_length


    def _process_instance(self, img_path, formula_id):
        # formula
        formula = self.form_prepro(self.formulas[int(formula_id)])

        # image
        img = imread(self.dir_images + "/" + img_path)
        img = self.img_prepro(img)

        return img, formula



    def __iter__(self):
        """
        Iterator over Dataset
        Yields:
            tuple of
                img: array
                formula: one formula
            or list of those tuples
        """
        n_iter = 0
        # instances is a tuple (img path, formula id) or a list of those
        for example in self.data_generator:
            if self.max_iter is not None and n_iter >= self.max_iter:
                break

            # just one training instance per line
            if type(example) is tuple:
                img_path, formula_id = example
                img, formula = self._process_instance(img_path, formula_id)

                if self.iter_mode == "data":
                    result = (img, formula)
                elif self.iter_mode == "full":
                    result = (img, formula, img_path, formula_id)

                # filter on the formula length
                if self.max_len is not None and len(formula) > self.max_len:
                    continue

            # multiple instances of the same example per line
            elif type(example) is list:
                result = []
                for instance in example:
                    img_path, formula_id = instance
                    img, formula = self._process_instance(img_path, formula_id)

                    if self.iter_mode == "data":
                        result.append((img, formula))
                    elif self.iter_mode == "full":
                        result.append((img, formula, img_path, formula_id))

                # filter on the first formula length
                if self.max_len is not None and len(result[0][1]) > self.max_len:
                    continue

            n_iter += 1
            yield result


    def __len__(self):
        if self.length is None:
            print("First call to len(dataset) - may take a while.")
            counter = 0
            for _ in self:
                counter += 1
            self.length = counter
            print("- done.")

        return self.length



    def generate_from_formulas(self, single=True, quality=100, density=200):
        """
        Generate images from the formulas and writes the correspondance in a
        matchin file.
        TODO: separate the rendering and the matching.
        TODO: exploit parallelism to render the images.
        """
        if not os.path.exists(self.dir_images):
            os.makedirs(self.dir_images)

        with open(self.path_matching, "a") as f:
            example = []
            for idx, formula in self.formulas.iteritems():

                # new example if 1. empty formula or 2. single mode
                if (single or (not single and len(formula) == 0)) and len(example) > 0:
                    f.write(" ".join(example) + "\n")
                    example = []

                elif len(formula) > 0:
                    try:
                        img_path = convert_to_png(formula, self.dir_images, idx, quality, density)
                        if img_path[-4:] == ".png":
                            example.append("{} {}".format(img_path, idx))

                    except Exception, e:
                        print e
