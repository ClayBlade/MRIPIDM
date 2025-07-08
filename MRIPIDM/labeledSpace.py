#Only in slicer cli
#only works once slicer is running???

import slicer
import slicer.util
import numpy as np

segmentation_path = r"D:/Users/clayt/NotDownloads/3d medical Diffusion data/MR Brain Segmentation Challenge 2018 Data/test/test/2/segm.nii.gz"  # Change to your file path

print("Loading segmentation...")
sucess, segNode = slicer.util.loadSegmentation(segmentation_path, returnNode=True)

if not sucess:
    raise RuntimeError("Failed to load segmentation.")

print("Converting to labelmap volume...")
labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "TempLabelmap")
slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(segNode, labelmapVolumeNode)

print("Extracting voxel data...")
label_array = slicer.util.arrayFromVolume(labelmapVolumeNode)
print("Label matrix shape:", label_array.shape)
print("Unique labels:", np.unique(label_array))
np.save(r"D:/Projects/MRIPIDM/output/labeledSpace/extracted_labels.npy", label_array)
