import kagglehub
import os

print("Downloading emmarex/plantdisease")
path1 = kagglehub.dataset_download("emmarex/plantdisease")
print("Path 1:", path1)

print("Downloading anshulm257/rice-disease-dataset")
path2 = kagglehub.dataset_download("anshulm257/rice-disease-dataset")
print("Path 2:", path2)

print("Downloading khanaamer/wheat-leaf-disease-dataset")
path3 = kagglehub.dataset_download("khanaamer/wheat-leaf-disease-dataset")
print("Path 3:", path3)
