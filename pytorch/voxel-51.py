import fiftyone.zoo as foz

dataset_name = "coco-2017"

dataset = foz.load_zoo_dataset("coco-2017", split="validation", max_samples=50)

