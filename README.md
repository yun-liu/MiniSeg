## [MiniSeg: An Extremely Minimum Network for Efficient COVID-19 Segmentation](https://yun-liu.github.io/papers/(AAAI'2021)MiniSeg%20-%20An%20Extremely%20Minimum%20Network%20for%20Efficient%20COVID-19%20Segmentation.pdf)

The rapid spread of the new pandemic, i.e., COVID-19, has severely threatened global health. Deep-learning-based computer-aided screening, e.g., COVID-19 infected CT area segmentation, has attracted much attention. However, the publicly available COVID-19 training data are limited, easily causing overfitting for traditional deep learning methods that are usually data-hungry with millions of parameters. On the other hand, fast training/testing and low computational cost are also necessary for quick deployment and development of COVID-19 screening systems, but traditional deep learning methods are usually computationally intensive. To address the above problems, we propose MiniSeg, a lightweight deep learning model for efficient COVID-19 segmentation. Compared with traditional segmentation methods, MiniSeg has several significant strengths: i) it only has 83K parameters and is thus not easy to overfit; ii) it has high computational efficiency and is thus convenient for practical deployment; iii) it can be fast retrained by other users using their private COVID-19 data for further improving performance. In addition, we build a comprehensive COVID-19 segmentation benchmark for comparing MiniSeg to traditional methods.

### Citations

If you are using the code/model/data provided here in a publication, please consider citing:

    @inproceedings{qiu2021miniseg,
      title={Mini{S}eg: An Extremely Minimum Network for Efficient {COVID}-19 Segmentation},
      author={Qiu, Yu and Liu, Yun and Li, Shijie and Xu, Jing},
      booktitle={AAAI Conference on Artificial Intelligence},
      year={2021}
    }

### Precomputed segmentation maps

We adopt a 5-fold cross validation to evaluate the proposed MiniSeg. The precomputed segmentation maps of five folds on four datasets are provided in the [`SegMaps`](https://github.com/yun-liu/MiniSeg/tree/master/SegMaps) folder.

### Testing and training

Before running the code, you should first put the images, masks and data lists to corresponding datasets folder. For examples, for COVID-19-CT100 dataset, the images are put in the /datasets/COVID-19-CT100/tr_im folder, the masks are put in the /datasets/COVID-19-CT100/tr_mask folde, and the tranining/testing data lists are put in the folder of /datasets/COVID-19-CT100/dataList.

#### Testing MiniSeg

#### Training MiniSeg

### Pretrained models

The pretrained models of five folds on four datasets are provided in the [`Pretrained`](https://github.com/yun-liu/MiniSeg/tree/master/Pretrained) folder.

### Running the code

We use Python 3.5, PyTorch 0.4.1, cuda 9.0, and numpy 1.17.3 to test the code. The `train.py` script is for training, and the `test.py` script is for testing.
