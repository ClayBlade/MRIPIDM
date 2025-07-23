#Only in slicer cli
#running this file clones itself into main dir 
#Delete this file and place in utils


import slicer
import slicer.util
import numpy as np
import os
import pickle

os.makedirs("D:/Projects/MRIPIDMoutput/ParametricMaps", exist_ok=True)


for folder in os.listdir(r"D:/Users/clayt/NotDownloads/3d medical Diffusion data/MR Brain Segmentation Challenge 2018 Data/test/test"):
    segmentation_path = rf"D:/Users/clayt/NotDownloads/3d medical Diffusion data/MR Brain Segmentation Challenge 2018 Data/test/test/{folder}/segm.nii.gz"

    success, segNode = slicer.util.loadSegmentation(segmentation_path, returnNode=True)

    if not success:
        print("error")
        raise RuntimeError("Failed to load segmentation.")

    labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "TempLabelmap")
    slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(segNode, labelmapVolumeNode)

    label_array = slicer.util.arrayFromVolume(labelmapVolumeNode)

    print("Label matrix shape:", label_array.shape)
    print("Unique labels:", np.unique(label_array))

    zSize = label_array.shape[0]
    ySize = label_array.shape[1]
    xSize = label_array.shape[2]

    Mx = np.zeros((zSize, ySize, xSize), dtype=np.float32)
    My = np.zeros((zSize, ySize, xSize), dtype=np.float32)
    Mz = np.ones((zSize, ySize, xSize), dtype=np.float32)

    T1 = np.zeros((zSize, ySize, xSize), dtype=np.float32)
    T2 = np.zeros((zSize, ySize, xSize), dtype=np.float32)
    T2star = np.zeros((zSize, ySize, xSize), dtype=np.float32)
    Rho = np.zeros((zSize, ySize, xSize), dtype=np.float32)

    for idx, label in np.ndenumerate(label_array):
        if label == 1: #Grey Matter
            T1[idx] = 0.95
            T2[idx] = 0.1
            T2star[idx] = 0.05
            Rho[idx] = 0.8
        elif label == 3: #White Matter
            T1[idx] = 0.6 
            T2[idx] = 0.08
            T2star[idx] = 0.04
            Rho[idx] = 0.65
        elif label == 5 or label == 6 or label == 4 or label == 2 or label == 7 or label == 8: #CSF
            T1[idx] = 4.5
            T2[idx] = 2.2
            T2star[idx] = 1.1
            Rho[idx] = 1
        else: #Background
            T1[idx] = 0.05
            T2[idx] = 0.05
            T2star[idx] = 0.025
            Rho[idx] = 0.5

    
    outpath = f"D:/Projects/MRIPIDMoutput/ParametricMaps/{folder}.npz"
    
    np.savez_compressed(outpath,
                        Gyro=4257.59,
                        xSize=xSize,
                        ySize=ySize,
                        zSize=zSize,
                        Mx=Mx,
                        My=My,
                        Mz=Mz,
                        T1=T1,
                        T2=T2,
                        T2star=T2star,
                        Rho=Rho)


    slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
    slicer.mrmlScene.RemoveNode(segNode)

for folder in os.listdir(r"D:/Users/clayt/NotDownloads/3d medical Diffusion data/MR Brain Segmentation Challenge 2018 Data/training/training"):
    segmentation_path = rf"D:/Users/clayt/NotDownloads/3d medical Diffusion data/MR Brain Segmentation Challenge 2018 Data/training/training/{folder}/segm.nii.gz"
    success, segNode = slicer.util.loadSegmentation(segmentation_path, returnNode=True)

    if not success:
        raise RuntimeError("Failed to load segmentation.")

    labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "TempLabelmap")
    slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(segNode, labelmapVolumeNode)

    label_array = slicer.util.arrayFromVolume(labelmapVolumeNode)

    print("Label matrix shape:", label_array.shape)
    print("Unique labels:", np.unique(label_array))
    
    zSize = label_array.shape[0]
    ySize = label_array.shape[1]
    xSize = label_array.shape[2]

    Mx = np.zeros((zSize, ySize, xSize), dtype=np.float32)
    My = np.zeros((zSize, ySize, xSize), dtype=np.float32)
    Mz = np.ones((zSize, ySize, xSize), dtype=np.float32)

    T1 = np.zeros((zSize, ySize, xSize), dtype=np.float32)
    T2 = np.zeros((zSize, ySize, xSize), dtype=np.float32)
    T2star = np.zeros((zSize, ySize, xSize), dtype=np.float32)
    Rho = np.zeros((zSize, ySize, xSize), dtype=np.float32)

    for idx, label in np.ndenumerate(label_array):
        if label == 1: #Grey Matter
            T1[idx] = 0.95
            T2[idx] = 0.1
            T2star[idx] = 0.05
            Rho[idx] = 0.8
        elif label == 3: #White Matter
            T1[idx] = 0.6 
            T2[idx] = 0.08
            T2star[idx] = 0.04
            Rho[idx] = 0.65
        elif label == 5 or label == 6 or label == 4 or label == 2 or label == 7 or label == 8: #CSF
            T1[idx] = 4.5
            T2[idx] = 2.2
            T2star[idx] = 1.1
            Rho[idx] = 1
        else: #Background
            T1[idx] = 0.05
            T2[idx] = 0.05
            T2star[idx] = 0.025
            Rho[idx] = 0.5

    outpath = f"D:/Projects/MRIPIDMoutput/ParametricMaps/{folder}.npz"
    
    np.savez_compressed(outpath,
                        Gyro=4257.59,
                        xSize=xSize,
                        ySize=ySize,
                        zSize=zSize,
                        Mx=Mx,
                        My=My,
                        Mz=Mz,
                        T1=T1,
                        T2=T2,
                        T2star=T2star,
                        Rho=Rho)

    slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
    slicer.mrmlScene.RemoveNode(segNode)
