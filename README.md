# Deep-MSVDD-PyTorch

This repository aims at providing a PyTorch implementation of DeepMSVDD as well as a simple comparison between this
state-of-the-art deep unsupervised AD method and the method which inspired it: DeepSVDD. Both methods 
are implemented using PyTorch, and many blocks of codes actually come from the original DeepSVDD paper code [repository](https://github.com/lukasruff/Deep-SVDD-PyTorch). 
A non-deep AD notebook is provided to help appreciate the actual contribution of deep learning. All non-deep methods are 
evaluated with and without a PCA dimensionality reduction as preprocessing. A CAE notebook achieving
AD using the reconstruction error is also provided for an additional comparison, and to provide pretrained weights for both DeepSVDD and DeepMSVDD. If you're looking for extra comparisons 
between shallow and deep AD methods on more realistic data, consider reading my papers [Deep Random Projection Outlyingness for Unsupervised Anomaly Detection](https://hal.archives-ouvertes.fr/hal-03203686) (cf. Table 8 in Appendix for the satellite dataset) 
and [From Unsupervised to Semi-supervised Anomaly Detection Methods for HRRP Targets](https://hal.archives-ouvertes.fr/hal-03254510) (AD on radar HRRPs, IEEE version of this paper [here](https://ieeexplore.ieee.org/abstract/document/9266497)).

### About DeepSVDD

papers http://proceedings.mlr.press/v80/ruff18a.html & https://arxiv.org/abs/1906.02694 

papers code https://github.com/lukasruff/Deep-SVDD-PyTorch & https://github.com/lukasruff/Deep-SAD-PyTorch

### About DeepMSVDD

paper https://epubs.siam.org/doi/pdf/10.1137/1.9781611976236.13

paper code https://github.com/zghafoori/Deep-Multi-Sphere-SVDD/tree/670ba3c7604347d249758b49f1865c51616c6a3c

### Regarding the deep AD notebooks (DeepCAE, DeepSVDD, DeepMSVDD)

- Batch size was set to 512, substantially higher than what's indicated in the paper (200) 
because it seems to stabilize test and validation AUCs during training (cf. graphs in notebooks). This isn't true for some of the difficult training
tasks considered in the results tables below. For instance, for the experiments where normality is MNIST [0,1,2,3,4,5,6] and FashionMNIST [0,1,2,3,4,5,6], the batch size is 
increased to 1024 to enable a smoother AUC evolution during training. Furthermore, for the experiment where normality is CIFAR10 [1,9], the batch size is also increased to 1024,
and the initial learning rate is set to 1e-3 instead of 1e-4 (the latter decreases during training as suggested in the DeepSVDD paper).
- The test AUC score is based on the best validation set performance epoch, as was done in https://arxiv.org/pdf/2001.08873.pdf.
- To enable "use_pretraining" for DeepSVDD and DeepMSVDD, set it to true and execute the DeepCAE notebook for the same number of seeds beforehand. The DeepCAE encoder trained
weights stored in ./trained_models will be used as the initialization, as indicated in the DeepSVDD paper.

### Possible differences with what's in the papers

Data preprocessing, best test epoch selection and weights initialization are not exactly the same. The best validation set epoch is used to choose the test AUC, 
and the test set may not stem from the same kind of split. We observed in several cases that the best validation epoch makes the final mean test AUC lower 
than a test AUC spike, and than a stable test AUC at the end of the learning epochs considered. However, this epoch selection process was kept because it is 
deemed fair. This is similar to
what was done in https://arxiv.org/pdf/2001.08873.pdf. One can check the evolution of valid and test AUCs during training on
graphs in the notebooks to observe this mismatch. The best test set epoch AUC is still recorded in last_results.txt (cf. the two last
cells of the neural networks notebooks).

Additionally, to avoid the issues associated with unbalanced datasets, the AD normal and anomalous classes are balanced 
in the validation and test sets. For any association of the dataset classes defining the normal AD class, we subsample the 
majority AD class (normal or anomalous) in order to reduce it to the same size as the minority AD class. One can note that AD 
performances do not strictly decrease when the normality complexity (# of normal classes) increases.

### Example results

Results are reported for 5 seeds (mean test AUCs +/- std). Each column is one experiment with its own normal classes (indicated in the top row), and each
row is associated with an AD method. "BTE" stands for best test epoch: this indicates performances where instead of the best validation set epoch, we use 
the best test set epoch to choose the test AUC.

|                                           | MNIST [0]         | MNIST [0,1,2]     | MNIST [0,1,2,3,4,5,6] |
|-------------------------------------------|-------------------|-------------------|-----------------------|
| **IF**                                    | 0.9636 +/- 0.0128 | 0.6664 +/- 0.0134 | 0.5030 +/- 0.0211     |
| **OC-SVM**                                | 0.9881 +/- 0.0022 | 0.8135 +/- 0.0042 | 0.5701 +/- 0.0049     |
| **LOF**                                   | 0.9960 +/- 0.0005 | 0.9655 +/- 0.0021 | too slow              |
| **IF (PCA)**                              | 0.9822 +/- 0.0056 | 0.8889 +/- 0.0100 | 0.6662 +/- 0.0162     |
| **OC-SVM (PCA)**                          | 0.9722 +/- 0.0039 | 0.7552 +/- 0.0044 | 0.5348 +/- 0.0056     |
| **LOF (PCA)**                             | 0.9938 +/- 0.0006 | 0.9625 +/- 0.0021 | 0.9137 +/- 0.0043     |
| **Deep CAE**                              | 0.9855 +/- 0.0044 | 0.9396 +/- 0.0086 | 0.7228 +/- 0.0191     |
| **Deep CAE - BTE**                        | 0.9855 +/- 0.0044 | 0.9396 +/- 0.0086 | 0.7229 +/- 0.0190     |
| **Deep SVDD**                             | 0.9729 +/- 0.0104 | 0.8809 +/- 0.0255 | 0.6814 +/- 0.0479     |
| **Deep SVDD - BTE**                       | 0.9736 +/- 0.0108 | 0.8811 +/- 0.0254 | 0.6818 +/- 0.0475     |
| **Deep SVDD (pretrained weights)**        | 0.9729 +/- 0.0104 | 0.8809 +/- 0.0255 | 0.6815 +/- 0.0479     |
| **Deep SVDD (pretrained weights) - BTE**  | 0.9736 +/- 0.0108 | 0.8811 +/- 0.0254 | 0.6818 +/- 0.0475     |
| **Deep MSVDD**                            | 0.9614 +/- 0.0142 | 0.8727 +/- 0.0137 | 0.6922 +/- 0.0374     |
| **Deep MSVDD - BTE**                      | 0.9750 +/- 0.0089 | 0.8919 +/- 0.0263 | 0.7246 +/- 0.0268     |
| **Deep MSVDD (pretrained weights)**       | 0.9615 +/- 0.0141 | 0.8691 +/- 0.0170 | 0.6700 +/- 0.0615     |
| **Deep MSVDD (pretrained weights) - BTE** | 0.9751 +/- 0.0089 | 0.8917 +/- 0.0261 | 0.7247 +/- 0.0268     |

|                                           | F-MNIST [0]       | F-MNIST [0,1,2]   | F-MNIST [0,1,2,3,4,5,6] |
|-------------------------------------------|-------------------|-------------------|-------------------------|
| **IF**                                    | 0.9076 +/- 0.0092 | 0.8238 +/- 0.0055 | 0.9047 +/- 0.0188       |
| **OC-SVM**                                | 0.8905 +/- 0.0059 | 0.7872 +/- 0.0087 | 0.8092 +/- 0.0069       |
| **LOF**                                   | 0.8611 +/- 0.0065 | 0.8656 +/- 0.0079 | too slow                |
| **IF (PCA)**                              | 0.9011 +/- 0.0076 | 0.8458 +/- 0.0092 | 0.8605 +/- 0.0114       |
| **OC-SVM (PCA)**                          | 0.9032 +/- 0.0055 | 0.7785 +/- 0.0092 | 0.8058 +/- 0.0062       |
| **LOF (PCA)**                             | 0.8670 +/- 0.0054 | 0.8578 +/- 0.0086 | 0.7641 +/- 0.0083       |
| **Deep CAE**                              | 0.9037 +/- 0.0075 | 0.8733 +/- 0.0049 | 0.8504 +/- 0.0263       |
| **Deep CAE - BTE**                        | 0.9049 +/- 0.0067 | 0.8741 +/- 0.0047 | 0.8504 +/- 0.0263       |
| **Deep SVDD**                             | 0.8858 +/- 0.0078 | 0.8580 +/- 0.0139 | 0.8654 +/- 0.0398       |
| **Deep SVDD - BTE**                       | 0.8894 +/- 0.0092 | 0.8594 +/- 0.0147 | 0.8656 +/- 0.0396       |
| **Deep SVDD (pretrained weights)**        | 0.8857 +/- 0.0078 | 0.8580 +/- 0.0139 | 0.8653 +/- 0.0399       |
| **Deep SVDD (pretrained weights) - BTE**  | 0.8894 +/- 0.0092 | 0.8594 +/- 0.0147 | 0.8654 +/- 0.0398       |
| **Deep MSVDD**                            | 0.8513 +/- 0.0198 | 0.8332 +/- 0.0097 | 0.8230 +/- 0.0381       |
| **Deep MSVDD - BTE**                      | 0.8521 +/- 0.0204 | 0.8365 +/- 0.0076 | 0.8474 +/- 0.0205       |
| **Deep MSVDD (pretrained weights)**       | 0.8378 +/- 0.0252 | 0.8361 +/- 0.0093 | 0.8307 +/- 0.0181       |
| **Deep MSVDD (pretrained weights) - BTE** | 0.8519 +/- 0.0209 | 0.8387 +/- 0.0082 | 0.8434 +/- 0.0177       |


|                                           | CIFAR10 [8]       | CIFAR10 [1,9]*    | CIFAR10 [2,3,4,5,6,7]** |
|-------------------------------------------|-------------------|-------------------|-------------------------|
| **IF**                                    | 0.6980 +/- 0.0217 | 0.4515 +/- 0.0107 | 0.6890 +/- 0.0083       |
| **OC-SVM**                                | 0.6454 +/- 0.0115 | 0.3891 +/- 0.0046 | 0.6640 +/- 0.0048       |
| **LOF**                                   | 0.6657 +/- 0.0159 | 0.3825 +/- 0.0058 | too slow                |
| **IF (PCA)**                              | 0.6590 +/- 0.0097 | 0.3129 +/- 0.0089 | 0.6184 +/- 0.0154       |
| **OC-SVM (PCA)**                          | 0.6316 +/- 0.0115 | 0.3988 +/- 0.0044 | 0.6621 +/- 0.0044       |
| **LOF (PCA)**                             | 0.6555 +/- 0.0156 | 0.3977 +/- 0.0056 | 0.5537 +/- 0.0080       |
| **Deep CAE**                              | 0.6865 +/- 0.0229 | 0.4115 +/- 0.0155 | 0.6563 +/- 0.0132       |
| **Deep CAE - BTE**                        | 0.6922 +/- 0.0179 | 0.4132 +/- 0.0153 | 0.6563 +/- 0.0132       |
| **Deep SVDD**                             | 0.7021 +/- 0.0437 | 0.6059 +/- 0.0222 | 0.7355 +/- 0.0511       |
| **Deep SVDD - BTE**                       | 0.7062 +/- 0.0392 | 0.6059 +/- 0.0222 | 0.7371 +/- 0.0514       |
| **Deep SVDD (pretrained weights)**        | 0.7021 +/- 0.0438 | 0.6059 +/- 0.0222 | 0.7355 +/- 0.0511       |
| **Deep SVDD (pretrained weights) - BTE**  | 0.7062 +/- 0.0392 | 0.6059 +/- 0.0222 | 0.7370 +/- 0.0514       |
| **Deep MSVDD**                            | 0.6961 +/- 0.0170 | 0.6143 +/- 0.0536 | 0.7388 +/- 0.0387       |
| **Deep MSVDD - BTE**                      | 0.6976 +/- 0.0158 | 0.6145 +/- 0.0534 | 0.7388 +/- 0.0387       |
| **Deep MSVDD (pretrained weights)**       | 0.6957 +/- 0.0172 | 0.6193 +/- 0.0614 | 0.7307 +/- 0.0315       |
| **Deep MSVDD (pretrained weights) - BTE** | 0.6971 +/- 0.0163 | 0.6193 +/- 0.0613 | 0.7307 +/- 0.0315       |

*This differs from the experiment mentioned in the Deep MSVDD paper, where the same classes define normality, but only a subset 
of the remaining classes define the anomalous AD class: "For the CIFAR10 dataset, vehicles, i.e., automobiles and trucks, were considered 
as the normal data and classes that represent animals were treated as anomalies.". This setup seems fairer since it respects 
one of the fundamental ideas of AD: anomalous data points are infinitely diverse.

**All animal classes are normal.

### To do

- fix seeds (performances are not exactly reproducible for the deep experiments, seeds are thus not correctly set everywhere they should)
- once seeds are fixed, generate example results on more (10 ?) seeds

### Disclaimer

The implementations in this repository have not been reviewed or validated by the authors of DeepSVDD and DeepMSVDD.
If you are familiar with these AD methods and notice an error in the implementations, feel free to create an issue.