import os
import random
import shutil
from itertools import islice

outputFolderPath = "Dataset/SplitData"
inputFolderPath = "Dataset/All"
splitRatio = {"train": 0.7, "val": 0.2, "test": 0.1}
classes = ["fake", "real"]

try:
    shutil.rmtree(outputFolderPath)
    # print("Removed Directory")
except OSError as e:
    os.mkdir(outputFolderPath)

# ------- Directories to Create ------- #
os.makedirs(f"{outputFolderPath}/train/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels", exist_ok=True)

# ------- Get Names ------- #
listNames = os.listdir(inputFolderPath)
# print(listNames)
# print(len(listNames))
uniqueNames = []
for name in listNames:
    uniqueNames.append(name.split(".")[0])
uniqueNames = list(set(uniqueNames))
# print(len(uniqueNames))

# ------- Shuffle ------- #
random.shuffle(uniqueNames)
# print(uniqueNames)

# ------- Find # of images for each folder ------- #
lenData = len(uniqueNames)
lenTrain = int(lenData * splitRatio["train"])
lenVal = int(lenData * splitRatio["val"])
lenTest = int(lenData * splitRatio["test"])
# print(f'Total Images: {lenData} \nSplit: {lenTrain, lenVal, lenTest}')

# ------- Put remainder images in Training ------- #
if lenData != lenTrain + lenVal + lenTest:
    remaining = lenData - (lenTrain + lenVal + lenTest)
    lenTrain += remaining
# print(f'Total Images: {lenData} \nSplit: {lenTrain, lenVal, lenTest}')

# ------- Split List ------- #
lengthToSplit = [lenTrain, lenVal, lenTest]
Input = iter(uniqueNames)
Output = [list(islice(Input, elem)) for elem in lengthToSplit]
# print(Output)
# print(len(Output))
print(f'Total Images: {lenData} \nSplit: {len(Output[0]), len(Output[1]), len(Output[2])}')

# ------- Copy Files ------- #
sequence = ["train", "val", "test"]
for i,out in enumerate(Output):
    for filename in out:
        for folder in sequence:
            shutil.copyfile(f'{inputFolderPath}/{filename}.jpg', f'{outputFolderPath}/{sequence[i]}/images/{filename}.jpg')
            shutil.copyfile(f'{inputFolderPath}/{filename}.txt', f'{outputFolderPath}/{sequence[i]}/labels/{filename}.txt')

print("Split Process Completed...")

# ------- Create Data.yaml ------- #
dataYaml = f'path: ../Data\n\
train: ../train/images\n\
val: ../val/images\n\
test: ../test/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}'

f = open(f"{outputFolderPath}/data.yaml", 'a')
f.write(dataYaml)
f.close()

print("Data.yaml file created...")

