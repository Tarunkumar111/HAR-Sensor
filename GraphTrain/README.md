Before running main.py file make sure the dataset folder should be in the following structure

dataset/
	DatasetName/ -->(e.g. WISDM/)
		raw/
			DatasetName_A.txt
			DatasetName_graph_indicator.txt
			DatasetName_graph_labels.txt
			DatasetName_node_attributes.txt
			DatasetName_node_labels.txt

When main.py is executed another folder named "process" is created which contains processed version of raw data in tensor format(.pt files) for faster execution

So the dataset directory get changed to

dataset/
	DatasetName/ -->(e.g. WISDM/)
		raw/
		processed/
			data.pt
			pre_filter.pt
			pre_transform.pt

