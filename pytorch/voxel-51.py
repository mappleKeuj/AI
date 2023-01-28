import fiftyone.zoo as foz

split = "validation"
dataset_name = "coco-2017"

dataset = foz.load_zoo_dataset(dataset_name, split=split)

