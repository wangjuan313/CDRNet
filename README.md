# CDRNet: Accurate Cup-to-Disc Ratio Measurement with Tight Bounding Box Supervision in Fundus Photography

This project hosts the codes for the implementation of **CDRNet: Accurate Cup-to-Disc Ratio Measurement with Tight Bounding Box Supervision in Fundus Photography Using Deep Learning** [[Journal]([CDRNet: accurate cup-to-disc ratio measurement with tight bounding box supervision in fundus photography using deep learning | SpringerLink](https://link.springer.com/article/10.1007/s11042-022-14183-2))] [[arXiv](https://arxiv.org/abs/2110.00943)].

## Dataset

All images are located in `data/glaucoma/images`. Three csv files, listing images for training, validation, and testing respectively, are in `data/csv` folder. For a csv file named as xxx.csv, there is a json file, named as annotation_xxx.json, which includes the bounding-box annotations for the images in the csv list. The json files locate in `data/csv` as well. 

Example of the structure of the folder for glaucoma dataset is as follows:

```bash
+ data
   + glaucoma
      + images
        - example1.png
        - example2.png
        - example3.png
        ...
      + csv
        - train.csv
        - annotation_train.json
        - validation.csv
        - annotation_validation.csv
        - testing.csv
        - annotation_testing.csv
```

## Training

```bash
#  The experiments include RetinaNet (exp_no=0), FSIS (exp_no=1), WSIS (exp_no=2,3), and CDRNet (exp_no=4,5,6,7)

# exp_no=0,1,2,3,4,5,6,7
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env tools/train_glaucoma.py --n_exp exp_no --world-size 4
```

## Evaluation and performance summary

```bash
# Validation and testing results for testing set and the dataset used in reader study, exp_no=0,1,2,3,4,5,6,7
CUDA_VISIBLE_DEVICES=0 python tools/eval_glaucomapy.py --n_exp exp_no

# performance summary
python tools/report_glaucoma.py
```

## Grader study

```bash
# Additional results for grader study
python tools/eval_glaucoma_reader_study.py

# performance summary
python tools/report_glaucoma_reader_study.py
```

> Note, this repository also includes implementation for the paper `Bounding Box Tightness Prior for Weakly Supervised Image Segmentation`. Please refer to [this link](https://github.com/wangjuan313/wsis-boundingbox) for more details. 

## Citations

Please consider citing our paper in your publications if the project helps your research.

```
@article{wang2022cdrnet,
  title={CDRNet: accurate cup-to-disc ratio measurement with tight bounding box supervision in fundus photography using deep learning},
  author={Wang, Juan and Xia, Bin},
  journal={Multimedia Tools and Applications},
  pages={1--23},
  year={2022},
  publisher={Springer}
}

@inproceedings{wang2021bounding,
  title={Bounding box tightness prior for weakly supervised image segmentation},
  author={Wang, Juan and Xia, Bin},
  booktitle={Medical Image Computing and Computer Assisted Intervention--MICCAI 2021: 24th International Conference, Strasbourg, France, September 27--October 1, 2021, Proceedings, Part II},
  pages={526--536},
  year={2021},
  organization={Springer}
}
```
