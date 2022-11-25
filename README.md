# Robust and Fast Measure of Information via Low-rank Representation

This repository contains the code used for the results reported in the paper 
Robust and Fast Measure of Information via Low-rank Representation (AAAI 2023).

Technical appendix can be found [here](LRMI_supp.pdf).

## Requirements

* Qt 5.x
* Visual Studio 2019
* Python 3.x

## Reproducing

To rerun the feature selection process, build the VS project and run it.

To reproduce the results in the paper:

Discrete datasets:
```
python3 feature/knn.py
```

Continuous datasets:
```
python3 feature/svm.py
```

## Cite

```
@inproceedings{dong2023robust,
  title={Robust and Fast Measure of Information via Low-rank Representation},
  author={Dong, Yuxin and Gong, Tieliang and Yu, Shujian and Chen, Hong and Li, Chen},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
```