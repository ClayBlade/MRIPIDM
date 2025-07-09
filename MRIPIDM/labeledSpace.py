#Only in slicer cli
#only works once slicer is running???

import slicer
import slicer.util
import numpy as np
import os
import pickle

for folder in os.listdir(r"D:/Users/clayt/NotDownloads/3d medical Diffusion data/MR Brain Segmentation Challenge 2018 Data/test/test"):
    segmentation_path = rf"D:/Users/clayt/NotDownloads/3d medical Diffusion data/MR Brain Segmentation Challenge 2018 Data/test/test/{folder}/segm.nii.gz"

    sucess, segNode = slicer.util.loadSegmentation(segmentation_path, returnNode=True)

    if not sucess:
        raise RuntimeError("Failed to load segmentation.")

    labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "TempLabelmap")
    slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(segNode, labelmapVolumeNode)

    label_array = slicer.util.arrayFromVolume(labelmapVolumeNode)

    print("Label matrix shape:", label_array.shape)
    print("Unique labels:", np.unique(label_array))

    zSize = label_array.shape[0]
    ySize = label_array.shape[1]
    xSize = label_array.shape[2]

    Mx = np.zeros((zSize, ySize, xSize), dtype=np.float32).flatten()
    My = np.zeros((zSize, ySize, xSize), dtype=np.float32).flatten()
    Mz = np.zeros((zSize, ySize, xSize), dtype=np.float32).flatten()
    

    T1 = np.zeros((zSize, ySize, xSize), dtype=np.float32).flatten()
    T2 = np.zeros((zSize, ySize, xSize), dtype=np.float32).flatten()
    T2star = np.zeros((zSize, ySize, xSize), dtype=np.float32).flatten()
    Rho = np.zeros((zSize, ySize, xSize), dtype=np.float32).flatten()

    for i, label in enumerate(label_array.flatten()):
        if label == 1: #Grey Matter
            T1[i] = 0.95
            T2[i] = 0.1
            T2star[i] = 0.05
            Rho[i] = 0.8
        elif label == 3: #White Matter
            T1[i] = 0.6 
            T2[i] = 0.08
            T2star[i] = 0.04
            Rho[i] = 0.65
        elif label == 5 or label == 6 or label == 4 or label == 2 or label == 7 or label == 8: #CSF
            T1[i] = 4.5
            T2[i] = 2.2
            T2star[i] = 1.1
            Rho[i] = 1
        else: #Background
            T1[i] = 0.05
            T2[i] = 0.05
            T2star[i] = 0.025
            Rho[i] = 0.5

    data_obj = {
        "Gyro" : 4257.59, #MHz/T
        "xSize" : xSize,
        "ySize" : ySize,
        "zSize" : zSize,
        "Mx" : Mx,
        "My" : My,     
        "Mz" : Mz,
        "T1" : T1,
        "T2" : T2,
        "T2star" : T2star,
        "Rho" : Rho
    }

    with open(rf"D:/Projects/MRIPIDM/output/labeledSpace/{folder}.pkl", "wb") as f:
        pickle.dump(data_obj, f)

for folder in os.listdir(r"D:/Users/clayt/NotDownloads/3d medical Diffusion data/MR Brain Segmentation Challenge 2018 Data/training/training"):
    segmentation_path = rf"D:/Users/clayt/NotDownloads/3d medical Diffusion data/MR Brain Segmentation Challenge 2018 Data/training/training/{folder}/segm.nii.gz"
    sucess, segNode = slicer.util.loadSegmentation(segmentation_path, returnNode=True)

    if not sucess:
        raise RuntimeError("Failed to load segmentation.")

    labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "TempLabelmap")
    slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(segNode, labelmapVolumeNode)

    label_array = slicer.util.arrayFromVolume(labelmapVolumeNode)

    print("Label matrix shape:", label_array.shape)
    print("Unique labels:", np.unique(label_array))
    
    zSize = label_array.shape[0]
    ySize = label_array.shape[1]
    xSize = label_array.shape[2]

    Mx = np.zeros((zSize, ySize, xSize), dtype=np.float32).flatten()
    My = np.zeros((zSize, ySize, xSize), dtype=np.float32).flatten()
    Mz = np.zeros((zSize, ySize, xSize), dtype=np.float32).flatten()

    T1 = np.zeros((zSize, ySize, xSize), dtype=np.float32).flatten()
    T2 = np.zeros((zSize, ySize, xSize), dtype=np.float32).flatten()
    T2star = np.zeros((zSize, ySize, xSize), dtype=np.float32).flatten()
    Rho = np.zeros((zSize, ySize, xSize), dtype=np.float32).flatten()

    for i, label in enumerate(label_array.flatten()):
        if label == 1: #Grey Matter
            T1[i] = 0.95
            T2[i] = 0.1
            T2star[i] = 0.05
            Rho[i] = 0.8
        elif label == 3: #White Matter
            T1[i] = 0.6 
            T2[i] = 0.08
            T2star[i] = 0.04
            Rho[i] = 0.65
        elif label == 5 or label == 6 or label == 4 or label == 2 or label == 7 or label == 8: #CSF
            T1[i] = 4.5
            T2[i] = 2.2
            T2star[i] = 1.1
            Rho[i] = 1
        else: #Background
            T1[i] = 0.05
            T2[i] = 0.05
            T2star[i] = 0.025
            Rho[i] = 0.5

    data_obj = {
        "Gyro" : 4257.59, #MHz/T
        "xSize" : xSize,
        "ySize" : ySize,
        "zSize" : zSize,
        "Mx" : Mx,
        "My" : My,     
        "Mz" : Mz,
        "T1" : T1,
        "T2" : T2,
        "T2star" : T2star,
        "Rho" : Rho
    }

    with open(rf"D:/Projects/MRIPIDM/output/labeledSpace/{folder}.pkl", "wb") as f:
        pickle.dump(data_obj, f)