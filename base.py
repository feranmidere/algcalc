import numpy as np
from dask import delayed

neg = float.__neg__  # setting negative function


def _product(*inputs):
    return np.product(inputs)


def _sum(*inputs):
    return sum(inputs)


def _divide(one, two):
    return one/two


def _turn_into_bracket(latex):  # Putting brackets around a latex string
    string = latex[1:-1]
    if string[0] + string[-1] == '()':
        bracket = string
    else:
        bracket = '(' + string + ')'
    return '{' + bracket + '}'


def _check_inv(child, type):  # Checking that an object is an additive inverse
    if type == 'neg':
        if isinstance(child, (int, float)):
            return child < 0
        elif isinstance(child, Variable):
            return False
        else:
            return child.inverted == type


def _get_latex_repr(obj):  # Getting the latex representation of an object
    if isinstance(obj, (int, float)):
        return '{' + str(obj) + '}'
    elif isinstance(obj, Variable):
        return '{' + obj.name + '}'
    else:
        return obj.latex_str


def _get_task(obj):  # Getting the task representation of an object
    if isinstance(obj, (int, float, str)):
        return obj
    elif isinstance(obj, Operation):
        return obj.node.create_task()
    elif isinstance(obj, Variable):
        return delayed(getattr)(obj, 'value')


class DelayedTaskNode:  # Class to represent tasks as nodes with children as paramters for a mathematical operation and a parent
    def __init__(self, func, args=None):
        self.func = func
        self.name = func.__name__
        self.parent = None
        self.children = list(args) if args is not None else None
        self.regulate_parenting()

    def regulate_parenting(self):
        for child in self.children:
            if isinstance(child, DelayedTaskNode):
                child.parent = self

    def create_task(self):
        if self.children is not None:
            task_args = map(_get_task, self.children)
            task = delayed(self.func)(*task_args)
        else:
            task = delayed(self.func)
        return task

    def get_level_list(self):
        level_list = [self.children]
        while self.children is not None:
            level_list.append(self.children)
        return level_list

    def vizualise(self):
        task = self.create_task()
        return task.visualize()

    def compute(self):
        task = self.create_task()
        return task.compute()

    def __str__(self, level=0):  # Stylish string representation of the node/tree
        ret = "\t"*level+repr(self.name)+"\n"
        for child in self.children:
            if isinstance(child, Operation):
                level = 0
                ret += child.node.__str__(level+1)
            elif isinstance(child, DelayedTaskNode):
                if not (child.name in ['__neg__', 'reciprocal']):
                    ret += child.__str__(level+1)
                else:
                    level += 1
                    ret += "\t"*level+child.__repr__()+"\n"
            else:
                ret += "\t"*(level+1)+child.__repr__()+"\n"
        return ret

    def __repr__(self):
        if self.children is not None:
            children = ', '.join(map(repr, self.children))
            text = self.name + f'({children})'
        else:
            text = self.name + '()'
        return text


class Operation:  # Base class for all mathematical operations
    def __init__(self, op, repr_str, latex_str, args, inverted=None):
        self.node = DelayedTaskNode(op, args)
        if args is not None:
            for arg in args:
                if hasattr(arg, 'node'):
                    arg.parent = self.node
        self.repr_str = repr_str
        self.latex_str = latex_str
        self.inverted = inverted

    def compute(self):
        return self.node.compute()

    def __neg__(self):
        repr_str = '-' + repr(self)
        latex_str = '{-' + _get_latex_repr(self) + '}'
        return Operation(float.__neg__, repr_str, latex_str, [self], inverted='neg')

    def reciprocal(self):
        return Fraction(1, self)

    def __repr__(self):
        return self.repr_str

    def _repr_latex_(self):
        return f'${self.latex_str}$'

    def __add__(self, other):
        return Sum(self, other)

    def __radd__(self, other):
        return Sum(other, self)

    def __sub__(self, other):
        return Sum(self, -other)

    def __rsub__(self, other):
        return Sum(other, -self)

    def __mul__(self, other):
        return Product(self, other)

    def __rmul__(self, other):
        return Product(other, self)

    def __truediv__(self, other):
        return Fraction(self, other)

    def __rtruediv__(self, other):
        return Fraction(other, self)

    def __pow__(self, other):
        return Power(self, other)

    def __rpow__(self, other):
        return Power(other, self)


class Sum(Operation):
    op = _sum

    def make_latex(input):
        if _check_inv(input, 'neg'):
            return _get_latex_repr(input)
        else:
            return f' + {_get_latex_repr(input)}'

    def __init__(self, *inputs):
        repr_str = ' + '.join(map(repr, inputs))
        latex_str = '{' + _get_latex_repr(inputs[0]) + ''.join([Sum.make_latex(input) for input in inputs[1:]]) + '}'
        super().__init__(Sum.op, repr_str, latex_str, inputs)

    def __neg__(self):
        repr_str = '-' + repr(self)
        latex_str = '{-(' + _get_latex_repr(self) + ')}'
        return Operation(float.__neg__, repr_str, latex_str, [self], inverted='neg')

    def __add__(self, other):
        added = other.node.children if isinstance(other, Sum) else [other]
        inputs = self.node.children + added
        return Sum(*inputs)

    def __sub__(self, other):
        inputs = self.node.children + [-other]
        return Sum(*inputs)


class Product(Operation):
    op = _product

    def latex_check_floatable(str):
        try:
            float(str.strip('{()}'))
            return True
        except ValueError:
            return False

    def latex_sort_key(latex):
        if Product.latex_check_floatable(latex):
            return 0
        if latex[1] == '-':
            return 1
        elif '\\frac' in latex:
            return 2
        elif latex[1] == '(' and latex[-2] == ')':
            return 4
        else:
            return 3

    def make_latex(inputs):
        latex_list = []
        for input in inputs:
            latex = _get_latex_repr(input)
            if isinstance(input, (Sum)):
                latex = _turn_into_bracket(latex)
            latex_list.append(latex)

        # This regulates the order that a product is written in - numbers first, then fractions and finally brackets
        sorted_latex_list = sorted(latex_list, key=Product.latex_sort_key)
        mapping = list(map(Product.latex_sort_key, sorted_latex_list))
        if mapping[0] == 0 and mapping[1] == 2:
            sorted_latex_list[1] = _turn_into_bracket(sorted_latex_list[1])
        return sorted_latex_list

    def __init__(self, *inputs):
        repr_str = ' * '.join(map(repr, inputs))
        latex_str = '{' + ''.join(Product.make_latex(inputs)) + '}'
        super().__init__(Product.op, repr_str, latex_str, inputs)

    def __mul__(self, other):
        added = other.node.children if isinstance(other, Product) else [other]
        inputs = self.node.children + added
        return Product(*inputs)


class Power(Operation):
    op = float.__pow__

    def __init__(self, *inputs):
        repr_str = '**'.join(map(repr, inputs))
        latex_str = '{' + _turn_into_bracket(_get_latex_repr(inputs[0])) + '^{' + _get_latex_repr(inputs[1]) + '}}'
        super().__init__(Power.op, repr_str, latex_str, inputs)


class Fraction(Operation):
    op = _divide

    def __init__(self, *inputs):
        repr_str = '{}/{}'.format(*map(repr, inputs))
        latex_str = '{\\frac{' + _get_latex_repr(inputs[0]) + '}{' + _get_latex_repr(inputs[1]) + '}}'
        super().__init__(Fraction.op, repr_str, latex_str, inputs)

    def reciprocal(self):
        numerator, denominator = self.node.children
        return Fraction(denominator, numerator)


class Variable:
    def __init__(self, name='x'):
        self.name = name
        self.value = None
        self.node = DelayedTaskNode(getattr, [self, 'value'])

    def set_value(self, value):
        self.value = float(value)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def _repr_latex_(self):
        return f'${self.name}$'

    def __neg__(self):
        repr_str = '-' + repr(self)
        latex_str = '{' + '-' + _get_latex_repr(self) + '}'
        return Operation(neg, repr_str, latex_str, args=[self], inverted='neg')

    def reciprocal(self):
        return Fraction(1, self)

    def __add__(self, other):
        return Sum(self, other)

    def __radd__(self, other):
        return Sum(other, self)

    def __sub__(self, other):
        return Sum(self, -other)

    def __rsub__(self, other):
        return Sum(other, -self)

    def __mul__(self, other):
        return Product(self, other)

    def __rmul__(self, other):
        return Product(other, self)

    def __truediv__(self, other):
        return Fraction(self, other)

    def __rtruediv__(self, other):
        return Fraction(other, self)

    def __pow__(self, other):
        return Power(self, other)

    def __rpow__(self, other):
        return Power(other, self)
