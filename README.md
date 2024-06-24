# Kmeans algorithm
This is a repo for learning about the kmeans clustering algorithm.
In depth information about the algorithm can be found on [this personal blog post.](https://medium.com/@dowra/a-clustering-algorithm-k-means-8b19e701a051)
The blog post can be used in conjunction with this repo to aid your curiosity and deepen our understanding.

# Package walkthrough:
### data
- contains data files
### images
- contains all generated graphs
### kmeans.py
- contains an implementation of the algorithm using scikit learn library
### kmean_native.py
- contains a native implementation of the algorithm, this is my personal attempt at using the pseudocode instruction in the blog to challenge myself to code this up. 
### kmeans_chatgpt.py
- contains an implementation from chatgpt

# Instructions on running the program:
1. Install all the libraries (`pip install x`) or use activate the virtual environment `source kmeans/bin/activate`
2. Run using `python3 file_name.py`


# Expected output from prepare_data() on kmeans.py
```
Shape of data: (150, 4)
Missing values:
sepal length (cm)    0
sepal width (cm)     0
petal length (cm)    0
petal width (cm)     0
dtype: int64
<bound method DataFrame.info of      
sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                  5.1               3.5                1.4               0.2
1                  4.9               3.0                1.4               0.2
2                  4.7               3.2                1.3               0.2
3                  4.6               3.1                1.5               0.2
4                  5.0               3.6                1.4               0.2
..                 ...               ...                ...               ...
145                6.7               3.0                5.2               2.3
146                6.3               2.5                5.0               1.9
147                6.5               3.0                5.2               2.0
148                6.2               3.4                5.4               2.3
149                5.9               3.0                5.1               1.8

[150 rows x 4 columns]>
Describe:
        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
count         150.000000        150.000000         150.000000        150.000000
mean            5.843333          3.057333           3.758000          1.199333
std             0.828066          0.435866           1.765298          0.762238
min             4.300000          2.000000           1.000000          0.100000
25%             5.100000          2.800000           1.600000          0.300000
50%             5.800000          3.000000           4.350000          1.300000
75%             6.400000          3.300000           5.100000          1.800000
max             7.900000          4.400000           6.900000          2.500000
Optimal k: 3
```

# Expected output from kmeans_native.py
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4000 entries, 0 to 3999
Data columns (total 3 columns):
 #   Column                Non-Null Count  Dtype  
---  ------                --------------  -----  
 0   id                    4000 non-null   int64  
 1   mean_dist_day         4000 non-null   float64
 2   mean_over_speed_perc  4000 non-null   int64  
dtypes: float64(1), int64(2)
memory usage: 93.9 KB
     mean_dist_day  mean_over_speed_perc
27       -0.566590              0.385136
336      -0.364768              1.625394
938      -0.546015             -0.417383
Training...
Iteration 1
Iteration 2
Iteration 3
Iteration 4
Iteration 5
Iteration 6
Iteration 7
Iteration 8
Iteration 9
Iteration 10

```

