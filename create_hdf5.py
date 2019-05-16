import cv2
import numpy as np
import h5py as h5
import os

def preprocessing(sar_group, opt_group):
    sar_post = sar_group[64:192, 64:192, :]
    opt_post = opt_group[64:192, 64:192, :]

    return  sar_post,opt_post


def read_images_sequentially(path_sar, path_sar_sal, path_opt, path_opt_sal):
    images_sar = os.listdir(path_sar)
    images_sar_sal = os.listdir(path_sar_sal)

    images_opt = os.listdir(path_opt)
    images_opt_sal = os.listdir(path_opt_sal)

    # foreach subfolder; open and save mutual images
    sarg_train = []
    optg_train = []

    sarg_test = []
    optg_test = []

    sarg_val = []
    optg_val = []

    labels_train = []
    labels_test = []
    labels_val = []


    counter = 0
    for i in range(len(images_sar)):

        s = cv2.imread(os.path.join(path_sar,images_sar[i]),0)
        s_s = cv2.imread(os.path.join(path_sar_sal,images_sar_sal[i]),0)

        o = cv2.imread(os.path.join(path_opt,images_opt[i]),0)
        o_s = cv2.imread(os.path.join(path_opt_sal,images_opt_sal[i]),0)

        sar_group = cv2.merge([s,s_s])
        opt_group = cv2.merge([o,o_s])

        sar_group,opt_group = preprocessing(sar_group,opt_group)

        if counter < 5:
            sarg_train.append(sar_group)
            optg_train.append(opt_group)

            labels_train.append((0,1))

        elif counter >=5 and counter <= 8:
            sarg_val.append(sar_group)
            optg_val.append(opt_group)

            labels_val.append((0,1))
        else:
            sarg_test.append(sar_group)
            optg_test.append(opt_group)

            labels_test.append((0,1))

        counter += 1

    return [sarg_train,optg_train,sarg_test,optg_test,sarg_val,optg_val, labels_train, labels_val, labels_test]


def save_h5(files, datas, labels):
    # Train Section
    s_tr = datas[0]
    o_tr = datas[1]

    # Validation Section
    s_v = datas[2]
    o_v = datas[3]

    # Test Section
    s_te = datas[4]
    o_te = datas[5]

    # Labels
    l_tr, l_v, l_te = labels[0], labels[1], labels[2]

    f_tr = files[0]
    f_v = files[1]
    f_te = files[2]

    s_tr_grp = f_tr.create_dataset("sar_group", data = s_tr)
    o_tr_grp = f_tr.create_dataset("optic_group", data = o_tr)
    l_tr_grp = f_tr.create_dataset("labels", data = l_tr)

    s_te_grp = f_te.create_dataset("sar_group", data = s_te)
    o_te_grp = f_te.create_dataset("optic_group", data = o_te )
    l_te_grp = f_te.create_dataset("labels", data = l_te)

    s_v_grp = f_v.create_dataset("sar_group", data = s_v)
    o_v_grp = f_v.create_dataset("optic_group", data = o_v)
    l_v_grp = f_v.create_dataset("labels", data = l_v)

    return True


if __name__=="__main__":

    # defition of images path
    sar_images_path = "./Dataset/sar_images"
    opt_images_path = "./Dataset/opt_images"
    sar_sal_images_path = "./Dataset/sar_sal"
    opt_sal_images_path = "./Dataset/opt_sal"


    # definition of h5 files
    hf_train = h5.File('Train_Matching.h5', 'w')
    hf_test = h5.File('Test_Matching.h5', 'w')
    hf_val = h5.File('Validation_Matching.h5', 'w')

    sarg_train,optg_train,sarg_test,optg_test,sarg_val,optg_val, l_train, l_val, l_test = \
        read_images_sequentially(sar_images_path,sar_sal_images_path,opt_images_path,opt_sal_images_path)

    # save them onto disk
    print("Lets save them onto hdf5  file")
    result = save_h5([hf_train,hf_val,hf_test], [sarg_train,optg_train,sarg_val,optg_val,sarg_test,optg_test] , [l_train, l_val, l_test])

    if result:
        print("Writing is Completed.")
    else:
        print("Check again writing stage!!!")


    #########
    # READING STAGE
    #########

    hf = h5.File("Train_Matching.h5", 'r')

    print(hf.keys())

    read_sar_tr = hf.get("sar_group")
    read_opt_tr = hf.get("optic_group")
    read_label_tr = hf.get("labels")

    print("Is equal Sar: ", np.array_equal(np.array(read_sar_tr), np.array(sarg_train)))
    print("Is equal Opt: ", np.array_equal(np.array(read_opt_tr), np.array(optg_train)))
    print("Is equal Labels: ", np.array_equal(np.array(read_label_tr), np.array(l_train)))

    ####################################################################################################################
    ############################################ CREATING H5 IS COMPLETED ##############################################
    ####################################################################################################################
