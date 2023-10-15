#!/bin/bash

while getopts 'e:c:i:l:w:t:n:d:' OPT; do
    case $OPT in
        e) exp=$OPTARG;;
        c) cuda=$OPTARG;;
		    i) identifier=$OPTARG;;
		    l) lr=$OPTARG;;
		    w) stu_w=$OPTARG;;
		    t) task=$OPTARG;;
		    n) num_epochs=$OPTARG;;
		    d) train_flag=$OPTARG;;
    esac
done
echo "exp:" $exp
echo "cuda:" $cuda
echo "num_epochs:" $num_epochs
echo "train_flag:" $train_flag



if [[ "${task}" == *"la"* ]];
  then
    label_ratio=$(echo "${task}" | cut -d'_' -f2)
    labeled_data="labeled_"${label_ratio}
    unlabeled_data="unlabeled_"${label_ratio}
    eval_data="eval_"${label_ratio}
    test_data="test"
    folder="Exp_SSL_LA_"${label_ratio}"/"
    if [ ${train_flag} = "true" ]; then
      python code/train_${exp}.py --exp ${folder}${exp}${identifier}/fold1 --seed 0 -g ${cuda} --base_lr ${lr} -w ${stu_w} -ep ${num_epochs} -sl ${labeled_data} -su ${unlabeled_data} -se ${eval_data} -t ${task}
    fi
    python code/test.py --exp ${folder}${exp}${identifier}/fold1 -g ${cuda} --split ${test_data} -t ${task}
    python code/evaluate_la.py --exp ${folder}${exp}${identifier} --folds 1 --split ${test_data} -t ${task}
fi

if [[ "${task}" == *"synapse"* ]];
  then
    label_ratio=$(echo "${task}" | cut -d'_' -f2)
    labeled_data="labeled_"${label_ratio}
    unlabeled_data="unlabeled_"${label_ratio}
    eval_data="eval"
    test_data="test"
    folder="Exp_IBSSL_Synapse_"${label_ratio}"/"
    if [ ${train_flag} = "true" ]; then
      python code/train_${exp}.py --exp ${folder}${exp}${identifier}/fold1 --seed 0 -g ${cuda} --base_lr ${lr} -w ${stu_w} -ep ${num_epochs} -sl ${labeled_data} -su ${unlabeled_data} -se ${eval_data} -t ${task}
    fi
    python code/test.py --exp ${folder}${exp}${identifier}/fold1 -g ${cuda} --split ${test_data} -t ${task}
    python code/evaluate.py --exp ${folder}${exp}${identifier} --folds 1 --split ${test_data} -t ${task}
    if [ ${train_flag} = "true" ]; then
      python code/train_${exp}.py --exp ${folder}${exp}${identifier}/fold2 --seed 1 -g ${cuda} --base_lr ${lr} -w ${stu_w} -ep ${num_epochs} -sl ${labeled_data} -su ${unlabeled_data} -se ${eval_data} -t ${task}
    fi
    python code/test.py --exp ${folder}${exp}${identifier}/fold2 -g ${cuda} --split ${test_data} -t ${task}
    python code/evaluate.py --exp ${folder}${exp}${identifier} --folds 2 --split ${test_data} -t ${task}
    if [ ${train_flag} = "true" ]; then
      python code/train_${exp}.py --exp ${folder}${exp}${identifier}/fold3 --seed 666 -g ${cuda} --base_lr ${lr} -w ${stu_w} -ep ${num_epochs} -sl ${labeled_data} -su ${unlabeled_data} -se ${eval_data} -t ${task}
    fi
    python code/test.py --exp ${folder}${exp}${identifier}/fold3 -g ${cuda} --split ${test_data} -t ${task}
    python code/evaluate.py --exp ${folder}${exp}${identifier} --folds 3 --split ${test_data} -t ${task}
fi



if [ ${task} = "mmwhs_ct2mr" ];
  then
    labeled_data="train_ct2mr_labeled"
    unlabeled_data="train_ct2mr_unlabeled"
    eval_data="eval_ct2mr"
    test_data="test_mr"
    modality="MR"
    folder="Exp_UDA_MMWHS_ct2mr/"

    if [ ${train_flag} = "true" ]; then
      python code/train_${exp}.py --exp ${folder}${exp}${identifier}/fold1 --seed 0 -g ${cuda} --base_lr ${lr} -w ${stu_w} -ep ${num_epochs} -sl ${labeled_data} -su ${unlabeled_data} -se ${eval_data} -t ${task}
    fi
    python code/test.py --exp ${folder}${exp}${identifier}/fold1 -g ${cuda} --split ${test_data} -t ${task}
    python code/evaluate.py --exp ${folder}${exp}${identifier} --folds 1  --split ${test_data} --modality ${modality} -t ${task}
fi


if [ ${task} = "mmwhs_mr2ct" ];
  then
    labeled_data="train_mr2ct_labeled"
    unlabeled_data="train_mr2ct_unlabeled"
    eval_data="eval_mr2ct"
    test_data="test_ct"
    modality="CT"
    folder="Exp_UDA_MMWHS_mr2ct/"
    if [ ${train_flag} = "true" ]; then
      python code/train_${exp}.py --exp ${folder}${exp}${identifier}/fold1 --seed 0 -g ${cuda} --base_lr ${lr} -w ${stu_w} -ep ${num_epochs} -sl ${labeled_data} -su ${unlabeled_data} -se ${eval_data} -t ${task}
    fi
    python code/test.py --exp ${folder}${exp}${identifier}/fold1 -g ${cuda} --split ${test_data} -t ${task}
    python code/evaluate.py --exp ${folder}${exp}${identifier} --folds 1 --split ${test_data} --modality ${modality} -t ${task}
fi


if [[ "${task}" == *"mnms"* ]];
  then
    domain=$(echo "${task}" | cut -d'_' -f2)
    label_ratio=$(echo "${task}" | cut -d'_' -f3)
    labeled_data="train_to${domain}_labeled_"${label_ratio}
    unlabeled_data="train_to${domain}_unlabeled_"${label_ratio}
    eval_data="test_to${domain}_"${label_ratio}
    test_data="test_to${domain}_"${label_ratio}
    folder="Exp_SemiDG_MNMs_to${domain}_"${label_ratio}"/"
    echo "cur_folder:" $folder
    if [ ${train_flag} = "true" ]; then
      python code/train_${exp}.py --exp ${folder}${exp}${identifier}/fold1 --seed 0 -g ${cuda} --base_lr ${lr} -w ${stu_w} -ep ${num_epochs} -sl ${labeled_data} -su ${unlabeled_data} -se ${eval_data} -t ${task} -cr 100
    fi
    if [[ "${exp}" == *"2d"* ]]; then
      python code/test_2d.py --exp ${folder}${exp}${identifier}/fold1 -g ${cuda} --split ${test_data} -t ${task}
      python code/evaluate_2d.py --exp ${folder}${exp}${identifier} --folds 1 --split ${test_data} -t ${task}
    else
      python code/test.py --exp ${folder}${exp}${identifier}/fold1 -g ${cuda} --split ${test_data} -t ${task}
      python code/evaluate.py --exp ${folder}${exp}${identifier} --folds 1 --split ${test_data} -t ${task}
    fi
fi


