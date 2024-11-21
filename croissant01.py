# testing CroissantBuilder

import os, sys, re 

import mlcroissant as mlc
import tensorflow_datasets as tdfs

kaggleURL = "https://www.kaggle.com/datasets/muhammadroshaanriaz/students-performance-dataset-cleaned/croissant/download"
huggingURL = "https://huggingface.co/api/datasets/fashion_mnist/croissant"
huggingURL2 = "https://huggingface.co/api/datasets/ReaKal/hg38.trf.short/croissant"

mydata = mlc.Dataset(kaggleURL).metadata.to_json()
print(mydata)

#sys.exit(0)

builder = tdfs.core.dataset_builders.CroissantBuilder(jsonld=huggingURL, record_set_ids=["fashion_mnist"], file_format="array_record")

builder.download_and_prepare()
#train, test = builder.as_data_source(split=['default[:80%]', 'default[80%:]'])

#print(len(train), len(test))
