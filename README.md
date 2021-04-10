# MIDAS-TASKS

The repository contains my solutions for the three sub-tasks in the Task 2 for MIDAS Summer Internship 2021.

I shall explain the main implementation idea in this file, the detailed solutions and explanations can be found on the respecitve readme files of the sub-tasks.
- [SUB TASK 1](SUB%20TASK%201/README.md)
- [SUB TASK 2](SUB%20TASK%202/README.md)
- [SUB TASK 3](SUB%20TASK%203/README.md)

For all the tasks I decided to implement the FaceNet paper for the following reasons:
- The dataset provided contained digits, lower-cased and upper-cased alphabets, some of which might look similar and hence give similar features. For example the numeral zero '0', upper case 'O' and lower case 'o' all look the same and might give similar features through a conventional CNN.
- The above problem can be easily beated by a neural network that can differentiate between images on the basis of their position on the Eucilidean or Cosine plane. This is where the FaceNet technique comes in.
- Using the FaceNet training method, I was able to map all of the given images to a 128 dimensional plane where the model tried to insert a margin of 0.2 units between the embeddings of each class.
- This ensures that even if two images have similar features, their embeddings on the euclidean space might lie apart given the specifics like height, widht and other smaller and finer details.
- Since I used the ```CosineSimilarity``` distance metric to train my model rather than the proposed Euclidean metric, even images that looked very similar on the Euclidean plane (like '0', 'O', and 'o') could be easily differentiated once the angular distance was introduced.
- The training architecture perfomed exceptionally well in the Sub Task 2, acheiving an **accuracy of 99.54% on the standard MNIST test dataset**.
