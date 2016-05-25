
import pandas as pd


class IncrementalModel:

    def fit(self, X, y, classes=None, sample_weight=None):
        pass

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        pass

    def predict(self, X):
        pass

class HigerLowerIncremental(IncrementalModel):
    """


    """
    def __init__(self, label_to_follow, steps=10):
        self.label_to_follow = label_to_follow
        self.steps = steps
        self.results = []
        self.values = []

    def fit(self, X):
        return self.partial_fit(X)

    def partial_fit(self, X):
        for i, row in X.iterrows():
            curr_val = row[self.label_to_follow]
            self.values.append(curr_val)

            new_vals = []
            for i in range(self.steps, 0, -1):
                value = None
                index = len(self.results) - i
                if index >= 0:
                    value = '1' if curr_val > self.values[index] else '0'

                new_vals.append(value)

            self.results.append(new_vals)
            print self.values
            print self.results

    def transform(self, X=None):
        col_names = [str(a) for a in range(self.steps)]
        result = pd.DataFrame(data=self.results, columns=col_names)
        return result


def iterate_rows(df, iterators):
    for i, row in df.iterate_rows():
        for module in iteraters:
            iterator.partial_fit(row)


if __name__ == '__main__':
    # import doctest
    # doctest.testmod()

    data_vals = [1, 2, 3, 2, 5, 4]
    data = pd.DataFrame(data=data_vals, columns=['price'])
    a = HigerLowerIncremental('price', 5)
    a.fit(data)
    print a.transform()