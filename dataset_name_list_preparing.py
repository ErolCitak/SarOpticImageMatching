import numpy
import os
import random

def preprocess_line(line):
    """
        sample input: datum/gray_fts/ROIs1970_fall_64_p694.npz
        desired output: ROIs1970_fall_s1_64_p694 and ROIs1970_fall_s2_64_p694
    """
    # remove redundant parts
    core_name = line.split('/')[2].rsplit('.')[0]
    core_name_parts = core_name.split("_")

    name_sar_parts = core_name_parts.copy()
    name_opt_parts = core_name_parts.copy()

    name_sar_parts.insert(2, 's1')
    name_opt_parts.insert(2, 's2')

    sar = '_'.join(name_sar_parts) + ".png"
    opt = '_'.join(name_opt_parts) + ".png"

    return  sar, opt


def save_txt(stage,sar_list, optic_list):
    with open('D:/sar_'+stage+'.txt', 'a') as the_file:
        for item in sar_list:
            the_file.write(item+'\n')

    with open('D:/optic_'+stage+'.txt', 'a') as the_file:
        for item in optic_list:
            the_file.write(item+'\n')

    return True



if __name__=="__main__":

    sar_general = []
    optic_general = []

    sar_pos_group = []
    sar_neg_group = []
    optic_pos_group = []
    optic_neg_group = []

    stage_name = "test"

    with open("./SEN12_Dataset_Lists/list."+stage_name+".txt", "r") as fileReader:
        for line in fileReader:

            # apply preprocessing onto raw line text
            sar_line, optic_line = preprocess_line(line)

            sar_pos_group.append(sar_line)
            optic_pos_group.append(optic_line)

    # now select randomly negative image for each sar image
    # seed = 999
    # 1 negative optic patch foreach sar image
    for i in range(len(sar_pos_group)):

        sar_neg_group.append(sar_pos_group[i])

        # select negative optic image patch
        random_index = 0
        while True:
            random_index = random.randint(0, len(sar_pos_group)-1)

            if i != random_index:
                break

        optic_neg_group.append(optic_pos_group[random_index])

    # Now positive and negative image group is concatenated into
    # one list, seperately

    for i in range(len(sar_pos_group)):
        s_pos = sar_pos_group[i]
        o_pos = optic_pos_group[i]

        s_neg = sar_neg_group[i]
        o_neg = optic_neg_group[i]

        sar_general.append(s_pos)
        sar_general.append(s_neg)

        optic_general.append(o_pos)
        optic_general.append(o_neg)

    # save them on txt file via line by line
    saving_res = save_txt(stage_name,sar_general, optic_general)

    print("Saving status:", saving_res)

