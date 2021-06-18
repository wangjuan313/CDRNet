import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
plt.close('all')

if __name__ == "__main__":
    dir_save_root = os.path.join('results', 'glaucoma_retinanet')
    graders = ['grader1', 'grader2', 'grader3', 'grader4']
    dataset_names = ['grader1', 'grader2', 'grader3', 'grader4']
    experiment_names = ['unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=6_iou_T=0.6',
                        # 'unet_det_mil_BB_10_dconv=6_explogs=8_L1sigma=6_iou_T=0.5'
                        ]
    # perf_array = {key: [] for key in dataset_names}
    cdr_array = np.zeros((5, 5, 3))
    dice_array = np.ones((5, 5, 3))
    experiment_name = experiment_names[0]
    output_dir = os.path.join(dir_save_root, experiment_name)
    print(f"<<< evaluating {experiment_name} ...")
    alg_cdr_array = []
    alg_dice_array = []
    cdr_keys = ['err_vcdr_det','err_hcdr_det', 'err_avg_det']
    dice_keys = ['cup_dice', 'disc_dice', 'avg_dice']
    for k, dataset_name in enumerate(dataset_names):
        data = pd.read_csv(os.path.join(output_dir, f"{dataset_name}_cdr.csv"))
        ## detection results
        keys = []
        data['err_vcdr_det'] = (data['gt_vcdr_bbox']-data['pd_vcdr_det']).abs()
        data['err_hcdr_det'] = (data['gt_hcdr_bbox']-data['pd_hcdr_det']).abs()
        data['err_avg_det'] = (data['err_vcdr_det'] + data['err_hcdr_det']) / 2.
        keys += ['err_vcdr_det', 'err_hcdr_det', 'err_avg_det', 'cup_error', 'disc_error']
        
        data['avg_dice'] = (data['cup_dice']+data['disc_dice'])/2
        keys += ['cup_dice', 'disc_dice', 'avg_dice']

        results = data[keys]
        mean = results.mean(axis=0).to_frame().T
        std  = results.std(axis=0).to_frame().T
        mean.rename(index={0: "mean"}, inplace=True)
        std.rename(index={0: "std"}, inplace=True)
        stat = pd.concat([mean, std], axis=0)
        # print(f"*** {dataset_name}")
        # print(stat)
        alg_cdr_array.append(mean[cdr_keys])
        alg_dice_array.append(mean[dice_keys])
        cdr_array[0, k+1, :] = mean[cdr_keys]
        cdr_array[k+1, 0, :] = mean[cdr_keys]
        dice_array[0, k+1, :] = mean[dice_keys]
        dice_array[k+1, 0, :] = mean[dice_keys]

    for k1 in range(len(graders)):
        grader1 = graders[k1]
        print(f">>> grader is {grader1} ...")
        grader_cdr_array = [alg_cdr_array[k1]]
        grader_dice_array = [alg_dice_array[k1]]
        for k2 in range(k1+1, len(graders)):
            grader2 = graders[k2]
            keys = []
            data = pd.read_csv(os.path.join(dir_save_root, f"dataset_{grader1}_{grader2}.csv"))
            data['err_vcdr_det'] = (data['vcdr_bbox1']-data['vcdr_bbox2']).abs()
            data['err_hcdr_det'] = (data['hcdr_bbox1']-data['hcdr_bbox2']).abs()
            data['err_avg_det'] = (data['err_vcdr_det'] + data['err_hcdr_det']) / 2.
            keys += ['err_vcdr_det', 'err_hcdr_det', 'err_avg_det']

            data['avg_dice'] = (data['cup_dice']+data['disc_dice'])/2
            keys += ['cup_dice', 'disc_dice', 'avg_dice']

            results = data[keys]
            mean = results.mean(axis=0).to_frame().T
            std  = results.std(axis=0).to_frame().T
            mean.rename(index={0: "mean"}, inplace=True)
            std.rename(index={0: "std"}, inplace=True)
            stat = pd.concat([mean, std], axis=0)
            # print(f"*** {grader2}")
            # print(stat)

            cdr_array[k1+1, k2+1, :] = mean[cdr_keys]
            cdr_array[k2+1, k1+1, :] = mean[cdr_keys]
            dice_array[k1+1, k2+1] = mean[dice_keys]
            dice_array[k2+1, k1+1] = mean[dice_keys]
    cdr_avg = cdr_array.sum(axis=0)/4
    dice_avg = (dice_array.sum(axis=0)-1)/4

    for k in range(len(cdr_keys)):
        print(f"<<< {cdr_keys[k]}")
        print(np.around(cdr_array[:, :, k], 4))
        print(np.around(cdr_avg[:, k], 4))
    for k in range(len(dice_keys)):
        print(f"<<< {dice_keys[k]}")
        print(np.around(dice_array[:, :, k], 3))
        print(np.around(dice_avg[:, k], 3))
