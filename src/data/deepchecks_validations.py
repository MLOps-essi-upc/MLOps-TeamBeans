
import os
import shutil
from datasets import load_dataset
from deepchecks.vision import classification_dataset_from_directory
from deepchecks.vision.suites import train_test_validation

# loads dataset for particular split
def loadDataSet(name, split):
    return load_dataset(name, split=split)

# return the path of the cache images for particular split
def pathToCopy(src, token):
    pathList = src.split(os.sep)
    result = []
    while len(pathList) > 0:
        if pathList[0] == token:
            break
        result.append(pathList.pop(0))
    
    return os.path.join('/', *result)

# copy images to data/raw
def copyToRaw(src, out):
    shutil.copytree(src, out, dirs_exist_ok=True)

# prepare data for the analysis
def prepareData(dsName, split, out):
    # load test data
    ds = loadDataSet(dsName, split)
    # get the path of the cashed images
    dsPath = pathToCopy(ds[0]["image_file_path"], split)
    print(dsPath)
    # copy test images from cache to raw folder
    copyToRaw(dsPath, out)

# conduct the analysis
def runTrainTestValidation(src):
    train, test = classification_dataset_from_directory( 
        root=src,
        object_type='VisionData', image_extension='jpg')
        
    suite = train_test_validation()
    result = suite.run(train_dataset=train, test_dataset=test)
    result.save_as_html('output.html')
    print("TEST DONE!")

# clear the data used for the analysis
def clearData(src):
    shutil.rmtree(src)

def main():
    datasetName ="beans"
    outPath = os.path.join('data', 'raw')
    prepareData(datasetName, 'train', outPath)
    prepareData(datasetName, 'test', outPath)
    runTrainTestValidation(outPath)
    clearData(outPath)


if __name__ == '__main__':
    main()

