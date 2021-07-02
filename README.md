# CDRNet: Accurate Cup-to-Disc Ratio Measurement with Tight Bounding Box Supervision in Fundus Photography

This project hosts the codes for the implementation of [CDRNet: Accurate Cup-to-Disc Ratio Measurement with Tight Bounding Box Supervision in Fundus Photography]() (under review).



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
@article{wang2021cdrnet,
  title     =  {CDRNet: Accurate Cup-to-Disc Ratio Measurement with Tight Bounding Box Supervision in Fundus Photography},
  author    =  {Wang, Juan and Xia, Bin},
  journal   =  {submitted to TMI},
  year      =  {2021}
}

@inproceedings{wang2021bounding,
  title     =  {Bounding Box Tightness Prior for Weakly Supervised Image Segmentation},
  author    =  {Wang, Juan and Xia, Bin},
  booktitle =  {24th International Conference on Medical Image Computing & Computer Assisted Intervention},
  year      =  {2021}
}
```



