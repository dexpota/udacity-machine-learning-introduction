# Model Evaluation Metrics

## TL;DR

We have a series of tools to create **prediction** and **regression** models
but how do we choose which one is better? We simply measure the performance of
each one and choose the best for the job. Those measuring tools are called
**model evaluation metrics**.


### Concepts

- Always split your data in two groups: the **training data** will be used to
  train your model, the **testing data**  will be used to evaluate your model;

- Never use your **testing data** to train your model!

- For a classification models we can build a **confusion matrix** based on the
  predictions made on some data. This matrix subdivide your data in **TP**,
  **TN**, **FN**, **FP**;

#### Metrics

- **Accuracy** or how many elements did we classify correctly? The answer is
  given by the ratio of the number of correctly classified points and the total
  classfied points;

  ```
  A = (TP + TN)/(All)
  ```

  - Accuracy is not always the best metric to choose. For example, let's say we
    have a problem where only a small percentage of the elements belong to a
    given class B and all other elements are A. Given a naive classifier that will
    classify all points as A will have a great accuracy, because of how the data is
    distributed;

- **Precision** or how many points did we classify correctly as positive out of
  all points classified positively?

  ```
  P = TP/(TP + FP)
  ```

- **Recall** or how many points did we classify correctly as positive out of
  all positive points?

  ```
  P = TP/(TP + FN)
  ```

- **F1 score** is the harmonic mean of **recall** and **accuracy**, this number
  will always be closer to the smaller number;

  ```
  F1 = 2 * P*R/(P + R)
  ```

- **Fbeta score** is

  ```
  Fb = (1 + b**2) * P*R/(b**2*P + R)
  ```

### ROC

## Notes

TODO: to be filled with notes

## Snippets

#### Split into testing and training data

```python
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
```

#### Compute accuracy

```python
from sklearn.metrics import accuracy_score
accuracy_score(y_true, y_pred)
```

### References

