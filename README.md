
# gGMMSP

This repository implements the superpixel algorithm of GMMSP whose serial implementation is located at  [GMMSP](https://github.com/ahban/GMMSP). 

The paper directly related to this repository is 

```bibtex
@article{Ban2018,
  author   =  "Ban, Zhihua and Liu, Jianguo and Fouriaux, Jeremy",
  title    =  "GMMSP on GPU",
  journal  =  "Journal of Real-Time Image Processing",
  year     =  "2018",
  month    =  "Mar",
  day      =  "17",
  issn     =  "1861-8219",
  doi      =  "10.1007/s11554-018-0762-3",
}
```



The above paper is used to describe this repository. The method it implemented is described in the following papers (they are the same method but one is peer-reviewed and another is for fast access to others).

```
@article{Ban18,
  author   =  {Zhihua Ban and Jianguo Liu and Li Cao},
  journal  =  {IEEE Transactions on Image Processing},
  title    =  {Superpixel Segmentation Using Gaussian Mixture Model},
  year     =  {2018},
  volume   =  {27},
  number   =  {8},
  pages    =  {4105-4117},
  doi      =  {10.1109/TIP.2018.2836306}
}
```



```
@article{Ban16,
  author    = {Zhihua Ban and Jianguo Liu and Li Cao},
  title     = {Superpixel Segmentation Using Gaussian Mixture Model},
  journal   = {{arXiv} preprint},
  volume    = {1612.08792},
  year      = {2016},
  url       = {http://arxiv.org/abs/1612.08792}
}
```




# Requirements

You have to make sure your PC meet the following requirments:
- At least one GPU that supports CUDA (e.g. GTX 1080, GTX 1070, GTX 1060)
- Compute Capabilities >= 3.0
- The lastest GPU driver
- Matlab Version >= R2015a


# Demo

Just run the demo script.

```matlab
> demo
```

The segmented result is shown below.

![](result/gvL.png)

# Help

If you want to know how to call the core function of `mx_gGMMSP`, just call it without any arguments.
```matlab
> mx_gGMMSP
```

# Contact

If you have any questions, please create an issue in this repository. I will soon give you a response to help you to run this code.
