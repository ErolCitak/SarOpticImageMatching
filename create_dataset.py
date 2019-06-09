import cv2
import numpy as np
import h5py as h5
import os
import datetime

def preprocessing(sar_group, opt_group):
    sar_post = sar_group[72:184, 72:184, :]
    opt_post = opt_group[72:184, 72:184, :]

    return  sar_post,opt_post


def reshape_data(data):
    data = np.reshape(data,(1,112,112))

    return data


def read_images_sequentially(path_sar, path_sar_sal, path_opt, path_opt_sal, sar_txt, opt_txt):
    # foreach subfolder; open and save mutual images
    sarg_pos_train = []
    sarg_sal_pos_train = []

    optg_pos_train = []
    optg_sal_pos_train = []

    sarg_neg_train = []
    sarg_sal_neg_train = []

    optg_neg_train = []
    optg_sal_neg_train = []

    labels_pos_train = []
    labels_neg_train = []

   #counter = 4
    indicator = 0
    for i in range(len(sar_txt)):

        #####################################################################
        #####################################################################
        #####################################################################

        sar_name_parts = sar_txt[i].lstrip("sar/").rstrip("\n").split("_")
        comparable_sar_name = '_'.join(sar_name_parts)

        sar_name_parts.insert(2,"s1")
        converted_sar_name =  '_'.join(sar_name_parts)


        opt_name_parts = opt_txt[i].lstrip("opt/").rstrip("\n").split("_")
        comparable_opt_name = '_'.join(opt_name_parts)

        opt_name_parts.insert(2, "s2")
        converted_opt_name = '_'.join(opt_name_parts)

        #####################################################################
        #####################################################################
        #####################################################################
        """
        converted_sar_name = sar_txt[i].rstrip("\n")
        converted_opt_name = opt_txt[i].rstrip("\n")

        sar_name_parts = converted_sar_name.split('_')
        opt_name_parts = converted_opt_name.split('_')

        del sar_name_parts[2]
        del opt_name_parts[2]

        comparable_sar_name = '_'.join(sar_name_parts)
        comparable_opt_name = '_'.join(opt_name_parts)
        """

        s = cv2.flip(cv2.imread(os.path.join(path_sar,converted_sar_name),0),1)
        s_s = cv2.flip(cv2.imread(os.path.join(path_sar_sal,converted_sar_name),0),1)

        o = cv2.flip(cv2.imread(os.path.join(path_opt,converted_opt_name),0),1)
        o_s = cv2.flip(cv2.imread(os.path.join(path_opt_sal,converted_opt_name),0),1)

        sar_group = cv2.merge([s,s_s])
        opt_group = cv2.merge([o,o_s])

        sar_group,opt_group = preprocessing(sar_group,opt_group)

        s, ss = cv2.split(sar_group)
        o, oss = cv2.split(opt_group)

        """
        s = reshape_data(s)
        ss = reshape_data(ss)
        o = reshape_data(o)
        oss = reshape_data(oss)
        """

        if comparable_opt_name == comparable_sar_name:
            # means this sample is positive sample == [0,1]
            sarg_pos_train.append(s)
            optg_pos_train.append(o)

            sarg_sal_pos_train.append(ss)
            optg_sal_pos_train.append(oss)

            labels_pos_train.append((0, 1))

        else:
            # means this sample is NOT positive sample == [1,0]
            sarg_neg_train.append(s)
            optg_neg_train.append(o)

            sarg_sal_neg_train.append(ss)
            optg_sal_neg_train.append(oss)

            labels_neg_train.append((1,0))

        if indicator % 1000 == 0:
            print("--> ", indicator)

        indicator += 1



    return [sarg_pos_train,optg_pos_train,sarg_neg_train,optg_neg_train,labels_pos_train,labels_neg_train, sarg_sal_pos_train, optg_sal_pos_train, sarg_sal_neg_train, optg_sal_neg_train]


def save_h5(files, datas, labels):
    # Train Section
    s_pos_tr = datas[0]
    o_pos_tr = datas[1]

    s_neg_tr = datas[2]
    o_neg_tr = datas[3]

    s_s_pos_tr = datas[4]
    o_s_pos_tr = datas[5]

    s_s_neg_tr = datas[6]
    o_s_neg_tr = datas[7]

    # Labels
    l_pos_tr = labels[0]
    l_neg_tr = labels[1]


    # File Info.
    f_tr = files[0]

    _ = f_tr.create_dataset("sar_pos_group", data = s_pos_tr)
    _ = f_tr.create_dataset("optic_pos_group", data = o_pos_tr)
    _ = f_tr.create_dataset("labels_pos", data = l_pos_tr)

    _ = f_tr.create_dataset("sar_neg_group", data = s_neg_tr)
    _ = f_tr.create_dataset("optic_neg_group", data = o_neg_tr)
    _ = f_tr.create_dataset("labels_neg", data = l_neg_tr)


    _ = f_tr.create_dataset("sar_sal_pos_group", data=s_s_pos_tr)
    _ = f_tr.create_dataset("optic_sal_pos_group", data=o_s_pos_tr)

    _ = f_tr.create_dataset("sar_sal_neg_group", data=s_s_neg_tr)
    _ = f_tr.create_dataset("optic_sal_neg_group", data=o_s_neg_tr)


    return True


if __name__=="__main__":
    print("Train Creation has started:" + str(datetime.datetime.now()))
    # txt files
    sar_lines = tuple(open("D:/Sen1_2/Lists/Train/list.long.supervised.sen12.sar.txt", 'r'))
    opt_lines = tuple(open("D:/Sen1_2/Lists/Train/list.long.supervised.sen12.opt.txt", 'r'))


    # defition of images path
    sar_images_path = "D:/Sen1_2_/SAR"
    opt_images_path = "D:/Sen1_2_/OPTIC"
    sar_sal_images_path = "D:/Sen1_2_/SAR_SAL"
    opt_sal_images_path = "D:/Sen1_2_/OPTIC_SAL"

    sarg_pos_train,optg_pos_train,sarg_neg_train,optg_neg_train,l_pos_train,l_neg_train, sarg_sal_pos_train, optg_sal_pos_train,\
    sarg_sal_neg_train, optg_sal_neg_train = read_images_sequentially(sar_images_path,sar_sal_images_path,opt_images_path,opt_sal_images_path, sar_lines, opt_lines)

    # definition of h5 files
    hf_train = h5.File('D:/Sen1_2_/Train_Matching_V2_V.h5', 'w')


    # save them onto disk
    print("Lets save them onto hdf5  file")
    result = save_h5([hf_train], [sarg_pos_train,optg_pos_train,sarg_neg_train,optg_neg_train, sarg_sal_pos_train, optg_sal_pos_train, sarg_sal_neg_train, optg_sal_neg_train ] ,
                     [l_pos_train,l_neg_train])

    if result:
        print("Writing is Completed.")
    else:
        print("Check again writing stage!!!")

    print("Train Creation has finished:" + str(datetime.datetime.now()))

    #########
    # HDF5 READING STAGE
    #########
    """
    hf = h5.File("Train_Matching.h5", 'r')

    print(hf.keys())

    read_sar_tr = hf.get("sar_group")
    read_opt_tr = hf.get("optic_group")
    read_label_tr = hf.get("labels")

    print("Is equal Sar: ", np.array_equal(np.array(read_sar_tr), np.array(sarg_train)))
    print("Is equal Opt: ", np.array_equal(np.array(read_opt_tr), np.array(optg_train)))
    print("Is equal Labels: ", np.array_equal(np.array(read_label_tr), np.array(l_train)))
    """
    ####################################################################################################################
    ############################################ CREATING H5 IS COMPLETED ##############################################
    ####################################################################################################################
