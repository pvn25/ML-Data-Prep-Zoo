## Benchmark Labeled Data


* data_train and data_test contains the base featurization split of the raw CSV files

* y_act column in the files denote the ground truth label. The coding of the labels is given as follows:

  Numeric : 0 <br />
  Categorical: 1 <br />
  Datetime:2 <br />
  Sentence:3 <br />
  URL: 4 <br />
  Numbers: 5 <br />
  List: 6 <br />
  Not-Generalizable: 7 <br />
  Custom Object (or Context-Specific): 8

* Metadata/ contains the record id and the source details of the raw CSV files.

* Our-Base-Featurization-Split/ contains the additional features extracted from the base featurized files for the ML models

* The raw data files that we used to create the base featurized files is available here for [download](https://drive.google.com/file/d/1ZPZY2wvDvsmnpQBABLz9ZyZRGvkEmo7B/view?usp=sharing).
