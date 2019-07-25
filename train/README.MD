

## epsilon-SVR with RBF

```
svm-train-gnu -s 3 -t 2 -g 0.1 -c 1 -e 0.0001 -h 0 train_dataset.dat model-eps.txt
```


## nu-SVR with RBF

```
svm-train-gnu -s 4 -t 2 -g 0.1 -c 1 -n 0.5 -e 0.0001 -h 0 train_dataset.dat model-nu.txt
```
