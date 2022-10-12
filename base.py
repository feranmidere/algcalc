from dask import delayed


class Variable:
    def __init__(self, name='x'):
        self.name = name
        self.value = None
        self.task = delayed(getattr)(self, 'value')

    def set_value(self, value):
        self.value = float(value)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def _repr_latex_(self):
        return f'${self.name}$'
