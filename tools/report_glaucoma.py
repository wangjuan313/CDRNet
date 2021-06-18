import os
import numpy as np
import pandas as pd

if __name__ == "__main__":
    dir_save_root = os.path.join('results', 'glaucoma_retinanet')
    dataset_names = ['valid', 'test']
    experiment_names = [#****** main results
                        'retinanet_B', 
                        'unet_BB_supervised', 
                        # 'unet_mil_BB', 'unet_mil_BB_10', 'unet_mil_BB_20', 'unet_mil_BB_30',
                        'unet_mil_BB_10_expsumr=8', #'unet_mil_BB_10_expsumr=6', 'unet_mil_BB_10_expsumr=10',
                        'unet_mil_BB_10_explogs=8', #'unet_mil_BB_10_explogs=6', 'unet_mil_BB_10_explogs=10',
                        'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=6_iou_T=0.6',
                        'unet_det_mil_BB_10_dconv=6_explogs=8_L1sigma=6_iou_T=0.5',
                        #****** sensitivity to T
                        'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=6',
                        'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=6_iou_T=0.5',
                        'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=6_iou_T=0.7',
                        #****** sensitivity to sigma
                        'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=3_iou_T=0.6',
                        'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=4_iou_T=0.6',
                        'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=5_iou_T=0.6',
                        'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=7_iou_T=0.6',
                        'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=8_iou_T=0.6',
                        # #****** sensitivity to T
                        # 'unet_det_mil_BB_10_dconv=6_explogs=8_L1sigma=6',
                        # 'unet_det_mil_BB_10_dconv=6_explogs=8_L1sigma=6_iou_T=0.6',
                        # 'unet_det_mil_BB_10_dconv=6_explogs=8_L1sigma=6_iou_T=0.7',
                        # #****** sensitivity to sigma
                        # 'unet_det_mil_BB_10_dconv=6_explogs=8_L1sigma=3_iou_T=0.5',
                        # 'unet_det_mil_BB_10_dconv=6_explogs=8_L1sigma=4_iou_T=0.5',
                        # 'unet_det_mil_BB_10_dconv=6_explogs=8_L1sigma=5_iou_T=0.5',
                        # 'unet_det_mil_BB_10_dconv=6_explogs=8_L1sigma=7_iou_T=0.5',
                        # 'unet_det_mil_BB_10_dconv=6_explogs=8_L1sigma=8_iou_T=0.5',                        
                        ]
    perf_array = {key: [] for key in dataset_names}
    for experiment_name in experiment_names:
        output_dir = os.path.join(dir_save_root, experiment_name)
        print(f"<<< evaluating {experiment_name} ...")
        for dataset_name in dataset_names:
            data = pd.read_csv(os.path.join(output_dir, f"{dataset_name}_cdr.csv"))
            ## detection results
            keys = []
            if ('retinanet' in experiment_name) | ('unet_det_mil' in experiment_name):
                data['err_vcdr_det'] = (data['gt_vcdr_bbox']-data['pd_vcdr_det']).abs()
                data['err_hcdr_det'] = (data['gt_hcdr_bbox']-data['pd_hcdr_det']).abs()
                data['err_avg_det'] = (data['err_vcdr_det'] + data['err_hcdr_det']) / 2.
                keys += ['err_vcdr_det', 'err_hcdr_det', 'err_avg_det', 'cup_error', 'disc_error']
                cc1 = np.corrcoef(data['gt_vcdr_bbox'], data['pd_vcdr_det'])
                cc2 = np.corrcoef(data['gt_hcdr_bbox'], data['pd_hcdr_det'])
                print(cc1[0, 1], cc2[0, 1])
            if ('retinanet_mil' in experiment_name) | ('unet_mil' in experiment_name) | \
                    ('supervised' in experiment_name) | ('unet_det_mil' in experiment_name):
                data['err_vcdr_seg'] = (data['gt_vcdr_bbox']-data['pd_vcdr_seg']).abs()
                data['err_hcdr_seg'] = (data['gt_hcdr_bbox']-data['pd_hcdr_seg']).abs()
                data['err_avg_seg'] = (data['err_vcdr_seg'] + data['err_hcdr_seg']) / 2.
                data['avg_dice'] = (data['cup_dice']+data['disc_dice'])/2
                if ('unet_mil' in experiment_name) | ('supervised' in experiment_name):
                    cc1 = np.corrcoef(data['gt_vcdr_bbox'], data['pd_vcdr_seg'])
                    cc2 = np.corrcoef(data['gt_hcdr_bbox'], data['pd_hcdr_seg'])
                    print(cc1[0, 1], cc2[0, 1])
                    keys += ['err_vcdr_seg', 'err_hcdr_seg', 'err_avg_seg', 'cup_dice', 'disc_dice', 'avg_dice']
                else:
                    keys += ['cup_dice', 'disc_dice', 'avg_dice']
            results = data[keys]
            mean = results.mean(axis=0).to_frame().T
            std  = results.std(axis=0).to_frame().T
            mean.rename(index={0: "mean"}, inplace=True)
            std.rename(index={0: "std"}, inplace=True)
            perf_array[dataset_name].append(mean)
            perf_array[dataset_name].append(std)

            stat = pd.concat([mean,std], axis=0)
            print(f"*** {dataset_name}")
            print(stat)