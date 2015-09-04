#### September 3, 2015
Try to improve speakerX, speakerY computation but didn't quite finish.

#### August 30, 2015
1. Build unoptimized classifier (linear SVM, also tried nonlinear kernels). Don't forget to normalize data!
2. Try adjusting class weights and optimizing parameters, but recall for test sets still only 64 for student, 
56 for silent, 54 for TA. Classifier not very good sadly :(

#### August 29, 2015
1. Explore the data using [this guide](https://jmetzen.github.io/2015-01-29/ml_advice.html).
1. David notices some instances that have unusually long duration (more than 100ms), and generates new filtered results.
I update my results with the new data, so remove observations about duration that I had before.
