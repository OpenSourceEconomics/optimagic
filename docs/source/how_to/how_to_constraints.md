(constraints)=

# How to specify constraints

## Constraints vs bounds

optimagic distinguishes between bounds and constraints. Bounds are lower and upper
bounds for parameters. In the literature, they are sometimes called box constraints.
Bounds are specified as `lower_bounds` and `upper_bounds` argument to `maximize` and
`minimize`.

Examples with bounds can be found in [this tutorial].

To specify more general constraints on your parameters, you can use the argument
`constraints`. The variety of constraints you can impose ranges from rather simple ones
(e.g. parameters are fixed to a value, a group of parameters is required to be equal) to
more complex ones (like general linear constraints, or even nonlinear constraints).

## Can you use constraints with all optimizers?

With the exception of general nonlinear constraints, we implement constraints via
reparametrizations. Details are explained [here]. This means that you can use all of the
constraints with any optimizer that supports bounds. Some constraints (e.g. fixing
parameters) can even be used with optimizers that do not support bounds.

## Example criterion function

Let's look at a variation of the sphere function to illustrate what kinds of constraints
you can impose and how you specify them in optimagic:

```{eval-rst}

.. code-block:: python

    >>> import numpy as np
    >>> import optimagic as om
    >>> def fun(params):
    ...     offset = np.linspace(1, 0, len(params))
    ...     x = params - offset
    ...     return x @ x

```

The unconstrained optimum of a six-dimensional version of this problem is:

```{eval-rst}

.. code-block:: python

    >>> res = om.minimize(
    ...    fun=fun,
    ...    params=np.array([2.5, 1, 1, 1, 1, -2.5]),
    ...    algorithm="scipy_lbfgsb",
    ... )
    >>> res.params.round(3)
    array([ 1. ,  0.8,  0.6,  0.4,  0.2, -0. ])

```

The unconstrained optimum is usually easy to see because all parameters enter the
criterion function in a additively separable way.

## Types of constraints

Below, we show a very simple example of each type of constraint implemented in
optimagic. For each constraint, we will select a subset of the parameters on which the
constraint is imposed via the `selector` argument, which is a function that takes in the
full parameter vector and returns the subset of parameters that should be constrained.

```{eval-rst}
.. dropdown:: fixed

    The simplest (but very useful) constraint fixes parameters at their start values.

    Let's take the above example and fix the first and last parameter to 2.5 and
    -2.5, respectively.

    .. code-block:: python

        >>> res = om.minimize(
        ...    fun=fun,
        ...    params=np.array([2.5, 1, 1, 1, 1, -2.5]),
        ...    algorithm="scipy_lbfgsb",
        ...    constraints=om.FixedConstraint(
        ...        selector=lambda params: params[[0, 5]]
        ...    ),
        ... )

    Looking at the optimization result, we get:

    >>> res.params.round(3)
    array([ 2.5,  0.8,  0.6,  0.4,  0.2, -2.5])

    Which is indeed the correct constrained optimum. Fixes are compatible with all
    optimizers.

```

```{eval-rst}
.. dropdown:: increasing

    In our unconstrained example, the optimal parameters are decreasing from left to
    right. Let's impose the constraint that the second, third and fourth parameter
    increase (weakly):

    .. code-block:: python


        >>> res = om.minimize(
        ...    fun=fun,
        ...    params=np.array([1, 1, 1, 1, 1, 1]),
        ...    algorithm="scipy_lbfgsb",
        ...    constraints=om.IncreasingConstraint(
        ...        selector=lambda params: params[[1, 2, 3]]
        ...    ),
        ... )


    Imposing the constraint on positions ``params[[1, 2, 3]]`` means that the parameter value
    at index position ``2`` has to be (weakly) greater than the value at position ``1``.
    Likewise, the parameter value at index position ``3`` has to be (weakly) greater than the
    value at position ``2``. Hence, imposing an increasing constraint with
    only one selected parameter has no effect. We need to specify at least two parameters to make
    a meaningful *relative* comparison.
    Note that the increasing constraint affect all three parameters, i.e. ``params[1]``,
    ``params[2]``, and ``params[3]`` because the optimal parameters in the unconstrained case
    are decreasing from left to right.

    Looking at the optimization result, we get:

    >>> res.params.round(3)
    array([1. , 0.6, 0.6, 0.6, 0.2, 0. ])

    Which is indeed the correct constrained optimum. Increasing constraints are only
    compatible with optimizers that support bounds.

```

```{eval-rst}
.. dropdown:: decreasing

    In our unconstrained example, the optimal parameters are decreasing from left to
    right already - without imposing any constraints. If we imposed an decreasing constraint
    without changing the order, it would simply have no effect.

    So let's impose one in a different order:

    .. code-block:: python

        >>> res = om.minimize(
        ...    fun=fun,
        ...    params=np.array([1, 1, 1, 1, 1, 1]),
        ...    algorithm="scipy_lbfgsb",
        ...    constraints=om.DecreasingConstraint(
        ...        selector=lambda params: params[[3, 0, 4]]
        ...    ),
        ... )

    Imposing the constraint on positions ``params[[3, 0, 4]]`` means that the parameter value
    at index position ``0`` has to be (weakly) smaller than the value at position ``3``.
    Likewise, the parameter value at index position ``4`` has to be (weakly) smaller than the
    value at position ``0``. Hence, imposing a decreasing constraint with
    only one selected parameter has no effect. We need to specify at least two parameters to make
    a meaningful *relative* comparison.
    Note that the decreasing constraint should have no effect on ``params[4]`` because it is
    smaller than the other two anyways in the unconstrained optimum, but it will change
    the optimal values of ``params[3]`` and ``params[0]``. Indeed we get:

    >>> res.params.round(3)
    array([ 0.7,  0.8,  0.6,  0.7,  0.2, -0. ])

    Which is the correct optimum. Decreasing constraints are only compatible with
    optimizers that support bounds.
```

```{eval-rst}
.. dropdown:: equality

    In our example, all optimal parameters are different. Let's constrain the first
    and last to be equal to each other:

    .. code-block:: python

        >>> res = om.minimize(
        ...    fun=fun,
        ...    params=np.array([1, 1, 1, 1, 1, 1]),
        ...    algorithm="scipy_lbfgsb",
        ...    constraints=om.EqualityConstraint(
        ...        selector=lambda params: params[[0, 5]]
        ...    ),
        ... )

    This yields:

    >>> res.params.round(3)
    array([0.5, 0.8, 0.6, 0.4, 0.2, 0.5])

    Which is the correct solution. Equality constraints are compatible with all
    optimizers.

```

```{eval-rst}
.. dropdown:: pairwise_equality

    Pairwise equality constraints are similar to equality constraints but impose that
    two or more groups of parameters are pairwise equal. Let's look at an example:

    .. code-block:: python

        >>> res = om.minimize(
        ...    fun=fun,
        ...    params=np.array([1, 1, 1, 1, 1, 1]),
        ...    algorithm="scipy_lbfgsb",
        ...    constraints=om.PairwiseEqualityConstraint(
        ...        selectors=[
        ...            lambda params: params[[0, 1]],
        ...            lambda params: params[[2, 3]]
        ...        ],
        ...    ),
        ... )



    This constraint imposes that ``params[0] == params[2]`` and
    ``params[1] == params[3]``. The optimal parameters with this constraint are:

    >>> res.params.round(3)
    array([ 0.8,  0.6,  0.8,  0.6,  0.2, -0. ])

```

```{eval-rst}
.. dropdown:: probability

    Let's impose the constraint that the first four parameters form valid
    probabilities, i.e. they should add up to one and be between zero and one.

    .. code-block:: python

        >>> res = om.minimize(
        ...    fun=fun,
        ...    params=np.array([0.3, 0.2, 0.25, 0.25, 1, 1]),
        ...    algorithm="scipy_lbfgsb",
        ...    constraints=om.ProbabilityConstraint(
        ...        selector=lambda params: params[:4]
        ...    ),
        ... )

    This yields again the correct result:

    .. code-block:: python

        >>> res.params.round(2) # doctest: +SKIP
        array([0.53, 0.33, 0.13, 0.  , 0.2 , 0.  ])


```

```{eval-rst}
.. dropdown:: covariance

    In many estimation problems, particularly when doing a maximum likelihood estimation,
    one has to estimate the covariance matrix of a random variable. The
    ``covariance`` costraint ensures that such a covariance matrix is always valid,
    i.e. positive semi-definite and symmetric. Due to its symmetry, only the lower
    triangle of a covariance matrix actually has to be estimated.

    Let's look at an example. We want to impose that the first three elements form the
    lower triangle of a valid covariance matrix.

    .. code-block:: python

        >>> res = om.minimize(
        ...    fun=fun,
        ...    params=np.ones(6),
        ...    algorithm="scipy_lbfgsb",
        ...    constraints=om.FlatCovConstraint(
        ...        selector=lambda params: params[:3]
        ...    ),
        ... )

    This yields the same solution as an unconstrained estimation because the constraint
    is not binding:

    >>> res.params.round(3)
    array([ 1.006,  0.784,  0.61 ,  0.4  ,  0.2  , -0.   ])

    We can now use one of optimagic's utility functions to actually build the covariance
    matrix out of the first three parameters:

    .. code-block:: python

        >>> from optimagic.utilities import cov_params_to_matrix
        >>> cov_params_to_matrix(res.params[:3]).round(2) # doctest: +NORMALIZE_WHITESPACE
        array([[1.01, 0.78],
               [0.78, 0.61]])


```

```{eval-rst}
.. dropdown:: sdcorr

    ``sdcorr`` constraints are very similar to ``covariance`` constraints. The only
    difference is that instead of estimating a covariance matrix, we estimate
    standard deviations and the correlation matrix of random variables.

    Let's look at an example. We want to impose that the first three elements form valid
    standard deviations and a correlation matrix.

    .. code-block:: python

        >>> res = om.minimize(
        ...    fun=fun,
        ...    params=np.ones(6),
        ...    algorithm="scipy_lbfgsb",
        ...    constraints=om.FlatSDCorrConstraint(
        ...        selector=lambda params: params[:3]
        ...    ),
        ... )


    This yields the same solution as an unconstrained estimation because the constraint
    is not binding:

    >>> res.params.round(3)
    array([ 1. ,  0.8,  0.6,  0.4,  0.2, -0. ])

    We can now use one of optimagic's utility functions to actually build the standard
    deviations and the correlation matrix:

    .. code-block:: python

        >>> from optimagic.utilities import sdcorr_params_to_sds_and_corr
        >>> sd, corr = sdcorr_params_to_sds_and_corr(res.params[:3])
        >>> sd.round(2)
        array([1. , 0.8])
        >>> corr.round(2) # doctest: +NORMALIZE_WHITESPACE
        array([[1. , 0.6],
               [0.6, 1. ]])


```

```{eval-rst}
.. dropdown:: linear

    Linear constraints are the most difficult but also the most powerful constraints
    in your toolkit. They can be used to express constraints of the form
    ``lower_bound <= weights.dot(x) <= upper_bound`` or
    ``weights.dot(x) = value`` where ``x`` are the selected parameters.

    Linear constraints have many of the other constraint types as special cases, but
    typically it is more convenient to use the special cases instead of expressing
    them as a linear constraint. Internally, it will make no difference.

    Let's impose the constraint that the average of the first four parameters is at
    least 0.95.

    .. code-block:: python

        >>> res = om.minimize(
        ...    fun=fun,
        ...    params=np.ones(6),
        ...    algorithm="scipy_lbfgsb",
        ...    constraints=om.LinearConstraint(
        ...        selector=lambda params: params[:4],
        ...        lower_bound=0.95,
        ...        weights=0.25,
        ...    ),
        ... )

    This yields:

    >>> res.params.round(2)
    array([ 1.25,  1.05,  0.85,  0.65,  0.2 , -0.  ])

    Where the first four parameters have an average of 0.95.

    In the above example, ``lower_bound`` and ``weights`` are scalars. They may, however,
    also be arrays (or even pytrees) with bounds and weights for each selected
    parameter.
```

```{eval-rst}
.. dropdown:: nonlinear

    .. warning::

        General nonlinear constraints that are specified via a black-box constraint
        function can only be used if you choose an optimizer that supports it.
        This feature is currently supported by the algorithms:

        * ``ipopt``
        * ``nlopt``: ``cobyla``, ``slsqp``, ``isres``, ``mma``
        * ``scipy``: ``cobyla``, ``slsqp``, ``trust_constr``

    You can use nonlinear constraints to express restrictions of the form
    ``lower_bound <= func(x) <= upper_bound`` or
    ``func(x) = value`` where ``x`` are the selected parameters and ``func`` is the
    constraint function.

    Let's impose the constraint that the product of all but the last parameter is 1.

    .. code-block:: python

        >>> res = om.minimize(
        ...    fun=fun,
        ...    params=np.ones(6),
        ...    algorithm="scipy_slsqp",
        ...    constraints=om.NonlinearConstraint(
        ...        selector=lambda params: params[:-1],
        ...        func=lambda x: np.prod(x),
        ...        value=1.0,
        ...    ),
        ... )

    This yields:

    >>> res.params.round(2)
    array([ 1.31,  1.16,  1.01,  0.87,  0.75, -0.  ])

    Where the product of all but the last parameters is equal to 1.

    If you have a function that calculates the derivative of your constraint, you can
    add this under the key `"derivative"` to the constraint dictionary. Otherwise,
    numerical derivatives are calculated for you if needed.

```

## Imposing multiple constraints at once

The above examples all just impose one constraint at a time. To impose multiple
constraints simultaneously, simple pass in a list of constraints. For example:

```{eval-rst}

.. code-block:: python

    >>> res = om.minimize(
    ...    fun=fun,
    ...    params=np.ones(6),
    ...    algorithm="scipy_lbfgsb",
    ...    constraints=[
    ...        om.EqualityConstraint(selector=lambda params: params[:2]),
    ...        om.LinearConstraint(
    ...            selector=lambda params: params[2:5],
    ...            weights=1,
    ...            value=3,
    ...        ),
    ...    ],
    ... )

    This yields:

    >>> res.params.round(2)
    array([0.9, 0.9, 1.2, 1. , 0.8, 0. ])

There are limits regarding the compatibility of overlapping constraints. You will
get a descriptive error message if your constraints are not compatible.

```

## How to select the parameters?

The parameters can be selected via a `selector` function. This function takes in the
full parameter vector and returns the subset of parameters that should be constrained.

Let's assume we have defined parameters in a nested dictionary:

```python
params = {"a": np.ones(2), "b": {"c": 3, "d": pd.Series([4, 5])}}
```

It is probably not a good idea to use a nested dictionary for so few parameters, but
let's ignore that.

Now assume we want to fix the parameters in the pandas Series at their start values. We
can do so as follows:

```python
res = om.minimize(
    fun=some_fun,
    params=params,
    algorithm="scipy_lbfgsb",
    constraints=om.FixedConstraint(selector=lambda params: params["b"]["d"]),
)
```

I.e. the value corresponding to `selector` is a python function that takes the full
`params` and returns a subset. The selected subset does not have to be a numpy array, it
can be an arbitrary pytree.

Using lambda functions if often convenient, but we could have just as well defined the
selector function using def.

```python
def my_selector(params):
    return params["b"]["d"]


res = om.minimize(
    fun=some_fun,
    params=params,
    algorithm="scipy_lbfgsb",
    constraints=om.FixedConstraint(selector=my_selector),
)
```

[here]: ../../explanation/implementation_of_constraints.md
[this tutorial]: ../tutorials/optimization_overview.ipynb
