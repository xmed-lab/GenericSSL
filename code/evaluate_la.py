import os
import numpy as np
import argparse
from medpy import metric
from tqdm import tqdm
from utils import read_list, read_nifti
from utils import config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default='synapse')
    parser.add_argument('--exp', type=str, default="fully")
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--modality', type=str, default='CT')

    args = parser.parse_args()

    config = config.Config(args.task)

    ids_list = read_list(args.split, task=args.task)
    results_all_folds = []

    txt_path = "./logs/"+args.exp+"/evaluation_res.txt"
    # print(txt_path)
    print("\n Evaluating...")
    fw = open(txt_path, 'w')
    for fold in range(1, args.folds+1):

        test_cls = [i for i in range(1, config.num_cls)]
        values = np.zeros((len(ids_list), len(test_cls), 4)) # dice and asd

        for idx, data_id in enumerate(tqdm(ids_list)):
            pred = read_nifti(os.path.join("./logs",args.exp, "fold"+str(fold), "predictions",f'{data_id}.nii.gz'))
            lb_path = os.path.join(config.save_dir, 'npy', f'{data_id}_label.npy')
            label = np.load(lb_path)

            padding_flag = label.shape[0] < config.patch_size[0] or label.shape[1] < config.patch_size[1] or label.shape[2] < config.patch_size[2]
            if padding_flag:
                pw = max((config.patch_size[0] - label.shape[0]) // 2 + 1, 0)
                ph = max((config.patch_size[1] - label.shape[1]) // 2 + 1, 0)
                pd = max((config.patch_size[2] - label.shape[2]) // 2 + 1, 0)
                # if padding_flag:
                label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)


            dd, ww, hh = label.shape

            for i in test_cls:
                pred_i = (pred == i)
                label_i = (label == i)
                if pred_i.sum() > 0 and label_i.sum() > 0:
                    dice = metric.binary.dc(pred == i, label == i) * 100
                    jaccard = metric.binary.jc(pred == i, label == i) * 100
                    hd95 = metric.binary.hd95(pred == i, label == i)
                    asd = metric.binary.asd(pred == i, label == i)
                    values[idx][i-1] = np.array([dice, jaccard, hd95, asd])
                elif pred_i.sum() > 0 and label_i.sum() == 0:
                    dice, jaccard, hd95, asd = 0, 0, 128, 128
                elif pred_i.sum() == 0 and label_i.sum() > 0:
                    dice, jaccard, hd95, asd = 0, 0, 128, 128
                elif pred_i.sum() == 0 and label_i.sum() == 0:
                    dice, jaccard, hd95, asd = 1, 1, 0, 0

                values[idx][i-1] = np.array([dice, jaccard, hd95, asd])

        values_mean_cases = np.mean(values, axis=0)
        results_all_folds.append(values)
        fw.write("Fold" + str(fold) + '\n')
        fw.write("------ Dice ------" + '\n')
        fw.write(str(np.round(values_mean_cases[:,0],1)) + '\n')
        fw.write("------ Jaccard ------" + '\n')
        fw.write(str(np.round(values_mean_cases[:,1],1)) + '\n')
        fw.write("------ HD95 ------" + '\n')
        fw.write(str(np.round(values_mean_cases[:,2],1)) + '\n')
        fw.write("------ ASD ------" + '\n')
        fw.write(str(np.round(values_mean_cases[:,3],1)) + '\n')
        fw.write('Average Dice:'+str(np.mean(values_mean_cases, axis=0)[0]) + '\n')
        fw.write('Average Jaccard:'+str(np.mean(values_mean_cases, axis=0)[1]) + '\n')
        fw.write('Average  HD95:'+str(np.mean(values_mean_cases, axis=0)[2]) + '\n')
        fw.write('Average  ASD:'+str(np.mean(values_mean_cases, axis=0)[3]) + '\n')
        fw.write("=================================")
        print("Fold", fold)
        print("------ Dice ------")
        print(np.round(values_mean_cases[:,0],1))
        print("------ Jaccard ------")
        print(np.round(values_mean_cases[:,1],1))
        print("------ HD95 ------")
        print(np.round(values_mean_cases[:,2],1))
        print("------ ASD ------")
        print(np.round(values_mean_cases[:,3],1))
        print(np.mean(values_mean_cases, axis=0)[0], np.mean(values_mean_cases, axis=0)[1])

    results_all_folds = np.array(results_all_folds)


    fw.write('\n\n\n')
    fw.write('All folds' + '\n')

    results_folds_mean = results_all_folds.mean(0)

    for i in range(results_folds_mean.shape[0]):
        fw.write("="*5 + " Case-" + str(ids_list[i]) + '\n')
        fw.write('\tDice:'+str(np.round(results_folds_mean[i][:,0],2).tolist()) + '\n')
        fw.write('\tJaccard:'+str(np.round(results_folds_mean[i][:,1],2).tolist()) + '\n')
        fw.write('\t HD95:'+str(np.round(results_folds_mean[i][:,2],2).tolist()) + '\n')
        fw.write('\t ASD:'+str(np.round(results_folds_mean[i][:,3],2).tolist()) + '\n')
        fw.write('\t'+'Average Dice:'+str(np.mean(results_folds_mean[i], axis=0)[0]) + '\n')
        fw.write('\t'+'Average Jaccard:'+str(np.mean(results_folds_mean[i], axis=0)[1]) + '\n')
        fw.write('\t'+'Average  HD95:'+str(np.mean(results_folds_mean[i], axis=0)[2]) + '\n')
        fw.write('\t'+'Average  ASD:'+str(np.mean(results_folds_mean[i], axis=0)[3]) + '\n')

    fw.write("=================================\n")
    fw.write('Final Dice of each class\n')
    fw.write(str([round(x,1) for x in results_folds_mean.mean(0)[:,0].tolist()]) + '\n')
    fw.write('Final Jaccard of each class\n')
    fw.write(str([round(x,1) for x in results_folds_mean.mean(0)[:,1].tolist()]) + '\n')
    fw.write('Final HD95 of each class\n')
    fw.write(str([round(x,1) for x in results_folds_mean.mean(0)[:,2].tolist()]) + '\n')
    fw.write('Final ASD of each class\n')
    fw.write(str([round(x,1) for x in results_folds_mean.mean(0)[:,3].tolist()]) + '\n')
    print("=================================")
    print('Final Dice of each class')
    print(str([round(x,1) for x in results_folds_mean.mean(0)[:,0].tolist()]))
    print('Final Jaccard of each class')
    print(str([round(x,1) for x in results_folds_mean.mean(0)[:,1].tolist()]))
    print('Final HD95 of each class')
    print(str([round(x,1) for x in results_folds_mean.mean(0)[:,2].tolist()]))
    print('Final ASD of each class')
    print(str([round(x,1) for x in results_folds_mean.mean(0)[:,3].tolist()]))
    std_dice = np.std(results_all_folds.mean(1).mean(1)[:,0])
    std_jacc = np.std(results_all_folds.mean(1).mean(1)[:,1])
    std_hd95 = np.std(results_all_folds.mean(1).mean(1)[:,2])
    std_asd = np.std(results_all_folds.mean(1).mean(1)[:,3])

    fw.write('Final Avg Dice: '+str(round(results_folds_mean.mean(0).mean(0)[0], 2)) +'±' +  str(round(std_dice,2)) + '\n')
    fw.write('Final Avg  Jaccard: '+str(round(results_folds_mean.mean(0).mean(0)[1], 2)) +'±' +  str(round(std_jacc,2)) + '\n')
    fw.write('Final Avg  HD95: '+str(round(results_folds_mean.mean(0).mean(0)[2], 2)) +'±' +  str(round(std_hd95,2)) + '\n')
    fw.write('Final Avg  ASD: '+str(round(results_folds_mean.mean(0).mean(0)[3], 2)) +'±' +  str(round(std_asd,2)) + '\n')

    print('Final Avg Dice: '+str(round(results_folds_mean.mean(0).mean(0)[0], 2)) +'±' +  str(round(std_dice,2)))
    print('Final Avg  Jaccard: '+str(round(results_folds_mean.mean(0).mean(0)[1], 2)) +'±' +  str(round(std_jacc,2)))
    print('Final Avg  HD95: '+str(round(results_folds_mean.mean(0).mean(0)[2], 2)) +'±' +  str(round(std_hd95,2)))
    print('Final Avg  ASD: '+str(round(results_folds_mean.mean(0).mean(0)[3], 2)) +'±' +  str(round(std_asd,2)))



