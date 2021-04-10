# Walk-through for Sub Task 1

## Introduction
The complete project can be replicated by following the [Jupyter Notebook](./subtask3Notebook.ipynb).
- I used the same process as discussed in [Sub Task 1](../SUB%20TASK%201/README.md) to train the network.
- I loaded the images using the Dataset class, transformed them into
  - single channel
  - resized them to (220,220)
  - split the validation to 15%
- I then trained the embedding network using this dataset for 80 epochs or around 16000 iterations. This was a time consuming process and took around 5 hours to train on the Tesla T4 GPU.
- The performance on validation set (extracted from the train set) was around 11% only.
- This embedding network was used to create a classifier which was trained on the provided dataset and tested on the MNIST test dataset.
- The test performance on the MNIST dataset is explained below:
  - Since the network learns to map embeddings rather than extracting features specifically, the random dataset provided ensured the detoriation of the model.
  - Each image is mapped on the angluar plane where the model tried to map images having the same label, since the label and images provided were random, the model tried to map random images.
  - These are the reasons why the model could not perform well on the MNIST dataset. 