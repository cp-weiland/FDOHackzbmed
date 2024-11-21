import os, sys, re, requests, json
import tensorflow_datasets as tfds

#jsonld="https://raw.githubusercontent.com/mlcommons/croissant/main/datasets/0.8/huggingface-mnist/metadata.json",
#jsonld="https://raw.githubusercontent.com/mlcommons/croissant/main/datasets/0.8/huggingface-mnist/metadata.json",

huggingURL = "https://huggingface.co/api/datasets/fashion_mnist/croissant"

localf = "myfoo.jsonld"

#response = requests.get(huggingURL, headers=None).json()
#with open(localf, "w") as f:
#  jsonld = json.dumps(response, indent=2)
#  f.write(jsonld)
#  print(jsonld)

#sys.exit(0)
  
builder = tfds.core.dataset_builders.CroissantBuilder(
    jsonld = huggingURL,
    record_set_ids=["fashion_mnist"],
    file_format='array_record',
)
builder.download_and_prepare()
ds = builder.as_data_source()
#print(ds['default'][0])

