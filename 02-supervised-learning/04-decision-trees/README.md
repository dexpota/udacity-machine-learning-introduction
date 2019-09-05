[decision-tree-classifier]: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

# Decision Tree

## Snippets

```python
model = DecisionTreeClassifier()
model.fit(X, y)
model.predict(x)
```

```python
# the maximum number of levels in the tree
max_depth = 7
# the minimum number of samples allowed in a leaf
min_samples_leaf = 10
# the minimum number of samples required to split an internal node
min_samples_split = 15

model = DecisionTreeClassifier(max_depth = 7, min_samples_leaf = 10, min_samples_split = 15)
```

## References

- [DecisionTreeClassifier][decision-tree-classifier];
