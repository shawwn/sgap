import numpy as np
import opt_einsum
from typing import Any, Sequence, FrozenSet, Optional, Tuple, Union, cast
import collections
import itertools

from jax import core
from jax import lax
from jax._src.numpy.lax_numpy import (
    _promote_dtypes,
    shape,
    reshape,
    _all,
    _removechars,
    _polymorphic_einsum_contract_path_handlers,
    dtypes,
    bool_,
)
from jax.util import unzip2


def einsum(*operands, out=None, optimize="greedy", precision=None, _use_xeinsum=False):
    if out is not None:
        raise NotImplementedError("The 'out' argument to jnp.einsum is not supported.")

    if (
        _use_xeinsum
        or isinstance(operands[0], str)
        and "{" in operands[0]
        and len(operands[1:]) == 2
    ):
        return lax.xeinsum(*operands)

    optimize = "greedy" if optimize is True else optimize
    # using einsum_call=True here is an internal api for opt_einsum

    # Allow handling of shape polymorphism
    non_constant_dim_types = {
        type(d)
        for op in operands
        if not isinstance(op, str)
        for d in np.shape(op)
        if not core.is_constant_dim(d)
    }
    if not non_constant_dim_types:
        einsum_contract_path_fn = opt_einsum.contract_path
    else:
        einsum_contract_path_fn = _polymorphic_einsum_contract_path_handlers[
            next(iter(non_constant_dim_types))
        ]
    operands, contractions = einsum_contract_path_fn(
        *operands, einsum_call=True, use_blas=True, optimize=optimize
    )

    contractions = tuple((a, frozenset(b), c) for a, b, c, *_ in contractions)
    return _einsum(operands, contractions, precision)


def _einsum(
    operands: Sequence,
    contractions: Sequence[Tuple[Tuple[int, ...], FrozenSet[str], str]],
    precision,
):
    operands = list(_promote_dtypes(*operands))

    def sum(x, axes):
        return lax.reduce(
            x,
            np.array(0, x.dtype),
            lax.add if x.dtype != bool_ else lax.bitwise_or,
            axes,
        )

    def sum_uniques(operand, names, uniques):
        if uniques:
            axes = [names.index(name) for name in uniques]
            operand = sum(operand, axes)
            names = _removechars(names, uniques)
        return operand, names

    def sum_repeats(operand, names, counts, keep_names):
        for name, count in counts.items():
            if count > 1:
                axes = [i for i, n in enumerate(names) if n == name]
                eye = lax._delta(operand.dtype, operand.shape, axes)
                if name not in keep_names:
                    operand = sum(operand * eye, axes)
                    names = names.replace(name, "")
                else:
                    operand = sum(operand * eye, axes[:-1])
                    names = names.replace(name, "", count - 1)
        return operand, names

    def filter_singleton_dims(operand, names, other_shape, other_names):
        s = shape(operand)
        new_shape = []
        new_names = []
        for i, d in enumerate(names):
            other_i = other_names.find(d)
            if (
                not core.symbolic_equal_dim(s[i], 1)
                or other_i == -1
                or core.symbolic_equal_dim(other_shape[other_i], 1)
            ):
                new_shape.append(s[i])
                new_names.append(d)
        return reshape(operand, tuple(new_shape)), "".join(new_names)

    for operand_indices, contracted_names_set, einstr in contractions:
        contracted_names = sorted(contracted_names_set)
        input_str, result_names = einstr.split("->")
        input_names = input_str.split(",")

        # switch on the number of operands to be processed in this loop iteration.
        # every case here sets 'operand' and 'names'.
        if len(operand_indices) == 1:
            operand = operands.pop(operand_indices[0])
            (names,) = input_names
            counts = collections.Counter(names)

            # sum out unique contracted indices with a single reduce-sum
            uniques = [name for name in contracted_names if counts[name] == 1]
            operand, names = sum_uniques(operand, names, uniques)

            # for every repeated index, do a contraction against an identity matrix
            operand, names = sum_repeats(operand, names, counts, result_names)

        elif len(operand_indices) == 2:
            lhs, rhs = map(operands.pop, operand_indices)
            lhs_names, rhs_names = input_names

            # handle cases where one side of a contracting or batch dimension is 1
            # but its counterpart is not.
            lhs, lhs_names = filter_singleton_dims(
                lhs, lhs_names, shape(rhs), rhs_names
            )
            rhs, rhs_names = filter_singleton_dims(
                rhs, rhs_names, shape(lhs), lhs_names
            )

            lhs_counts = collections.Counter(lhs_names)
            rhs_counts = collections.Counter(rhs_names)

            # sum out unique contracted indices in lhs and rhs
            lhs_uniques = [
                name
                for name in contracted_names
                if lhs_counts[name] == 1 and rhs_counts[name] == 0
            ]
            lhs, lhs_names = sum_uniques(lhs, lhs_names, lhs_uniques)

            rhs_uniques = [
                name
                for name in contracted_names
                if rhs_counts[name] == 1 and lhs_counts[name] == 0
            ]
            rhs, rhs_names = sum_uniques(rhs, rhs_names, rhs_uniques)

            # for every repeated index, contract against an identity matrix
            lhs, lhs_names = sum_repeats(
                lhs, lhs_names, lhs_counts, result_names + rhs_names
            )
            rhs, rhs_names = sum_repeats(
                rhs, rhs_names, rhs_counts, result_names + lhs_names
            )

            lhs_or_rhs_names = set(lhs_names) | set(rhs_names)
            contracted_names = [x for x in contracted_names if x in lhs_or_rhs_names]
            lhs_and_rhs_names = set(lhs_names) & set(rhs_names)
            batch_names = [x for x in result_names if x in lhs_and_rhs_names]

            lhs_batch, rhs_batch = unzip2(
                (lhs_names.find(n), rhs_names.find(n)) for n in batch_names
            )

            # NOTE(mattjj): this can fail non-deterministically in python3, maybe
            # due to opt_einsum
            assert _all(
                name in lhs_names
                and name in rhs_names
                and lhs.shape[lhs_names.index(name)] == rhs.shape[rhs_names.index(name)]
                for name in contracted_names
            )

            # contract using lax.dot_general
            batch_names_str = "".join(batch_names)
            lhs_cont, rhs_cont = unzip2(
                (lhs_names.index(n), rhs_names.index(n)) for n in contracted_names
            )
            deleted_names = batch_names_str + "".join(contracted_names)
            remaining_lhs_names = _removechars(lhs_names, deleted_names)
            remaining_rhs_names = _removechars(rhs_names, deleted_names)
            # Try both orders of lhs and rhs, in the hope that one of them means we
            # don't need an explicit transpose. opt_einsum likes to contract from
            # right to left, so we expect (rhs,lhs) to have the best chance of not
            # needing a transpose.
            names = batch_names_str + remaining_rhs_names + remaining_lhs_names
            if names == result_names:
                dimension_numbers = ((rhs_cont, lhs_cont), (rhs_batch, lhs_batch))
                # operand = lax.dot_general(rhs, lhs, dimension_numbers, precision)
                operand = dot_general(rhs, lhs, dimension_numbers)
            else:
                names = batch_names_str + remaining_lhs_names + remaining_rhs_names
                dimension_numbers = ((lhs_cont, rhs_cont), (lhs_batch, rhs_batch))
                # operand = lax.dot_general(lhs, rhs, dimension_numbers, precision)
                operand = dot_general(lhs, rhs, dimension_numbers)
        else:
            raise NotImplementedError  # if this is actually reachable, open an issue!

        # the resulting 'operand' with axis labels 'names' should be a permutation
        # of the desired result
        assert len(names) == len(result_names) == len(set(names))
        assert set(names) == set(result_names)
        if names != result_names:
            perm = tuple([names.index(name) for name in result_names])
            operand = lax.transpose(operand, perm)
        operands.append(operand)  # used in next iteration

    return operands[0]


def dot_general(lhs, rhs, dimension_numbers):
    (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
    new_id = itertools.count()
    lhs_axis_ids = [next(new_id) for _ in lhs.shape]
    rhs_axis_ids = [next(new_id) for _ in rhs.shape]
    lhs_out_axis_ids = lhs_axis_ids[:]
    rhs_out_axis_ids = rhs_axis_ids[:]

    for lhs_axis, rhs_axis in zip(lhs_contracting, rhs_contracting):
        shared_id = next(new_id)
        lhs_axis_ids[lhs_axis] = shared_id
        rhs_axis_ids[rhs_axis] = shared_id
        lhs_out_axis_ids[lhs_axis] = None
        rhs_out_axis_ids[rhs_axis] = None

    batch_ids = []
    for lhs_axis, rhs_axis in zip(lhs_batch, rhs_batch):
        shared_id = next(new_id)
        lhs_axis_ids[lhs_axis] = shared_id
        rhs_axis_ids[rhs_axis] = shared_id
        lhs_out_axis_ids[lhs_axis] = None
        rhs_out_axis_ids[rhs_axis] = None
        batch_ids.append(shared_id)

    not_none = lambda x: x is not None
    out_axis_ids = filter(not_none, batch_ids + lhs_out_axis_ids + rhs_out_axis_ids)
    assert lhs.dtype == rhs.dtype
    dtype = np.float32 if lhs.dtype == dtypes.bfloat16 else None
    # out = np.einsum(lhs, lhs_axis_ids, rhs, rhs_axis_ids, out_axis_ids, dtype=dtype)
    out = np_einsum(
        lhs, lhs_axis_ids, rhs, rhs_axis_ids, out_axis_ids, dtype=dtype, optimize=True
    )
    return out.astype(dtypes.bfloat16) if lhs.dtype == dtypes.bfloat16 else out


from numpy.core.einsumfunc import c_einsum, einsum_path, tensordot, asanyarray


def np_einsum(*operands, out=None, optimize=False, **kwargs):
    """
    einsum(subscripts, *operands, out=None, dtype=None, order='K',
           casting='safe', optimize=False)

    Evaluates the Einstein summation convention on the operands.

    Using the Einstein summation convention, many common multi-dimensional,
    linear algebraic array operations can be represented in a simple fashion.
    In *implicit* mode `einsum` computes these values.

    In *explicit* mode, `einsum` provides further flexibility to compute
    other array operations that might not be considered classical Einstein
    summation operations, by disabling, or forcing summation over specified
    subscript labels.

    See the notes and examples for clarification.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation as comma separated list of
        subscript labels. An implicit (classical Einstein summation)
        calculation is performed unless the explicit indicator '->' is
        included as well as subscript labels of the precise output form.
    operands : list of array_like
        These are the arrays for the operation.
    out : ndarray, optional
        If provided, the calculation is done into this array.
    dtype : {data-type, None}, optional
        If provided, forces the calculation to use the data type specified.
        Note that you may have to also give a more liberal `casting`
        parameter to allow the conversions. Default is None.
    order : {'C', 'F', 'A', 'K'}, optional
        Controls the memory layout of the output. 'C' means it should
        be C contiguous. 'F' means it should be Fortran contiguous,
        'A' means it should be 'F' if the inputs are all 'F', 'C' otherwise.
        'K' means it should be as close to the layout as the inputs as
        is possible, including arbitrarily permuted axes.
        Default is 'K'.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur.  Setting this to
        'unsafe' is not recommended, as it can adversely affect accumulations.

          * 'no' means the data types should not be cast at all.
          * 'equiv' means only byte-order changes are allowed.
          * 'safe' means only casts which can preserve values are allowed.
          * 'same_kind' means only safe casts or casts within a kind,
            like float64 to float32, are allowed.
          * 'unsafe' means any data conversions may be done.

        Default is 'safe'.
    optimize : {False, True, 'greedy', 'optimal'}, optional
        Controls if intermediate optimization should occur. No optimization
        will occur if False and True will default to the 'greedy' algorithm.
        Also accepts an explicit contraction list from the ``np.einsum_path``
        function. See ``np.einsum_path`` for more details. Defaults to False.

    Returns
    -------
    output : ndarray
        The calculation based on the Einstein summation convention.

    See Also
    --------
    einsum_path, dot, inner, outer, tensordot, linalg.multi_dot
    einops :
        similar verbose interface is provided by
        `einops <https://github.com/arogozhnikov/einops>`_ package to cover
        additional operations: transpose, reshape/flatten, repeat/tile,
        squeeze/unsqueeze and reductions.
    opt_einsum :
        `opt_einsum <https://optimized-einsum.readthedocs.io/en/stable/>`_
        optimizes contraction order for einsum-like expressions
        in backend-agnostic manner.

    Notes
    -----
    .. versionadded:: 1.6.0

    The Einstein summation convention can be used to compute
    many multi-dimensional, linear algebraic array operations. `einsum`
    provides a succinct way of representing these.

    A non-exhaustive list of these operations,
    which can be computed by `einsum`, is shown below along with examples:

    * Trace of an array, :py:func:`numpy.trace`.
    * Return a diagonal, :py:func:`numpy.diag`.
    * Array axis summations, :py:func:`numpy.sum`.
    * Transpositions and permutations, :py:func:`numpy.transpose`.
    * Matrix multiplication and dot product, :py:func:`numpy.matmul` :py:func:`numpy.dot`.
    * Vector inner and outer products, :py:func:`numpy.inner` :py:func:`numpy.outer`.
    * Broadcasting, element-wise and scalar multiplication, :py:func:`numpy.multiply`.
    * Tensor contractions, :py:func:`numpy.tensordot`.
    * Chained array operations, in efficient calculation order, :py:func:`numpy.einsum_path`.

    The subscripts string is a comma-separated list of subscript labels,
    where each label refers to a dimension of the corresponding operand.
    Whenever a label is repeated it is summed, so ``np.einsum('i,i', a, b)``
    is equivalent to :py:func:`np.inner(a,b) <numpy.inner>`. If a label
    appears only once, it is not summed, so ``np.einsum('i', a)`` produces a
    view of ``a`` with no changes. A further example ``np.einsum('ij,jk', a, b)``
    describes traditional matrix multiplication and is equivalent to
    :py:func:`np.matmul(a,b) <numpy.matmul>`. Repeated subscript labels in one
    operand take the diagonal. For example, ``np.einsum('ii', a)`` is equivalent
    to :py:func:`np.trace(a) <numpy.trace>`.

    In *implicit mode*, the chosen subscripts are important
    since the axes of the output are reordered alphabetically.  This
    means that ``np.einsum('ij', a)`` doesn't affect a 2D array, while
    ``np.einsum('ji', a)`` takes its transpose. Additionally,
    ``np.einsum('ij,jk', a, b)`` returns a matrix multiplication, while,
    ``np.einsum('ij,jh', a, b)`` returns the transpose of the
    multiplication since subscript 'h' precedes subscript 'i'.

    In *explicit mode* the output can be directly controlled by
    specifying output subscript labels.  This requires the
    identifier '->' as well as the list of output subscript labels.
    This feature increases the flexibility of the function since
    summing can be disabled or forced when required. The call
    ``np.einsum('i->', a)`` is like :py:func:`np.sum(a, axis=-1) <numpy.sum>`,
    and ``np.einsum('ii->i', a)`` is like :py:func:`np.diag(a) <numpy.diag>`.
    The difference is that `einsum` does not allow broadcasting by default.
    Additionally ``np.einsum('ij,jh->ih', a, b)`` directly specifies the
    order of the output subscript labels and therefore returns matrix
    multiplication, unlike the example above in implicit mode.

    To enable and control broadcasting, use an ellipsis.  Default
    NumPy-style broadcasting is done by adding an ellipsis
    to the left of each term, like ``np.einsum('...ii->...i', a)``.
    To take the trace along the first and last axes,
    you can do ``np.einsum('i...i', a)``, or to do a matrix-matrix
    product with the left-most indices instead of rightmost, one can do
    ``np.einsum('ij...,jk...->ik...', a, b)``.

    When there is only one operand, no axes are summed, and no output
    parameter is provided, a view into the operand is returned instead
    of a new array.  Thus, taking the diagonal as ``np.einsum('ii->i', a)``
    produces a view (changed in version 1.10.0).

    `einsum` also provides an alternative way to provide the subscripts
    and operands as ``einsum(op0, sublist0, op1, sublist1, ..., [sublistout])``.
    If the output shape is not provided in this format `einsum` will be
    calculated in implicit mode, otherwise it will be performed explicitly.
    The examples below have corresponding `einsum` calls with the two
    parameter methods.

    .. versionadded:: 1.10.0

    Views returned from einsum are now writeable whenever the input array
    is writeable. For example, ``np.einsum('ijk...->kji...', a)`` will now
    have the same effect as :py:func:`np.swapaxes(a, 0, 2) <numpy.swapaxes>`
    and ``np.einsum('ii->i', a)`` will return a writeable view of the diagonal
    of a 2D array.

    .. versionadded:: 1.12.0

    Added the ``optimize`` argument which will optimize the contraction order
    of an einsum expression. For a contraction with three or more operands this
    can greatly increase the computational efficiency at the cost of a larger
    memory footprint during computation.

    Typically a 'greedy' algorithm is applied which empirical tests have shown
    returns the optimal path in the majority of cases. In some cases 'optimal'
    will return the superlative path through a more expensive, exhaustive search.
    For iterative calculations it may be advisable to calculate the optimal path
    once and reuse that path by supplying it as an argument. An example is given
    below.

    See :py:func:`numpy.einsum_path` for more details.

    Examples
    --------
    >>> a = np.arange(25).reshape(5,5)
    >>> b = np.arange(5)
    >>> c = np.arange(6).reshape(2,3)

    Trace of a matrix:

    >>> np.einsum('ii', a)
    60
    >>> np.einsum(a, [0,0])
    60
    >>> np.trace(a)
    60

    Extract the diagonal (requires explicit form):

    >>> np.einsum('ii->i', a)
    array([ 0,  6, 12, 18, 24])
    >>> np.einsum(a, [0,0], [0])
    array([ 0,  6, 12, 18, 24])
    >>> np.diag(a)
    array([ 0,  6, 12, 18, 24])

    Sum over an axis (requires explicit form):

    >>> np.einsum('ij->i', a)
    array([ 10,  35,  60,  85, 110])
    >>> np.einsum(a, [0,1], [0])
    array([ 10,  35,  60,  85, 110])
    >>> np.sum(a, axis=1)
    array([ 10,  35,  60,  85, 110])

    For higher dimensional arrays summing a single axis can be done with ellipsis:

    >>> np.einsum('...j->...', a)
    array([ 10,  35,  60,  85, 110])
    >>> np.einsum(a, [Ellipsis,1], [Ellipsis])
    array([ 10,  35,  60,  85, 110])

    Compute a matrix transpose, or reorder any number of axes:

    >>> np.einsum('ji', c)
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> np.einsum('ij->ji', c)
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> np.einsum(c, [1,0])
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> np.transpose(c)
    array([[0, 3],
           [1, 4],
           [2, 5]])

    Vector inner products:

    >>> np.einsum('i,i', b, b)
    30
    >>> np.einsum(b, [0], b, [0])
    30
    >>> np.inner(b,b)
    30

    Matrix vector multiplication:

    >>> np.einsum('ij,j', a, b)
    array([ 30,  80, 130, 180, 230])
    >>> np.einsum(a, [0,1], b, [1])
    array([ 30,  80, 130, 180, 230])
    >>> np.dot(a, b)
    array([ 30,  80, 130, 180, 230])
    >>> np.einsum('...j,j', a, b)
    array([ 30,  80, 130, 180, 230])

    Broadcasting and scalar multiplication:

    >>> np.einsum('..., ...', 3, c)
    array([[ 0,  3,  6],
           [ 9, 12, 15]])
    >>> np.einsum(',ij', 3, c)
    array([[ 0,  3,  6],
           [ 9, 12, 15]])
    >>> np.einsum(3, [Ellipsis], c, [Ellipsis])
    array([[ 0,  3,  6],
           [ 9, 12, 15]])
    >>> np.multiply(3, c)
    array([[ 0,  3,  6],
           [ 9, 12, 15]])

    Vector outer product:

    >>> np.einsum('i,j', np.arange(2)+1, b)
    array([[0, 1, 2, 3, 4],
           [0, 2, 4, 6, 8]])
    >>> np.einsum(np.arange(2)+1, [0], b, [1])
    array([[0, 1, 2, 3, 4],
           [0, 2, 4, 6, 8]])
    >>> np.outer(np.arange(2)+1, b)
    array([[0, 1, 2, 3, 4],
           [0, 2, 4, 6, 8]])

    Tensor contraction:

    >>> a = np.arange(60.).reshape(3,4,5)
    >>> b = np.arange(24.).reshape(4,3,2)
    >>> np.einsum('ijk,jil->kl', a, b)
    array([[4400., 4730.],
           [4532., 4874.],
           [4664., 5018.],
           [4796., 5162.],
           [4928., 5306.]])
    >>> np.einsum(a, [0,1,2], b, [1,0,3], [2,3])
    array([[4400., 4730.],
           [4532., 4874.],
           [4664., 5018.],
           [4796., 5162.],
           [4928., 5306.]])
    >>> np.tensordot(a,b, axes=([1,0],[0,1]))
    array([[4400., 4730.],
           [4532., 4874.],
           [4664., 5018.],
           [4796., 5162.],
           [4928., 5306.]])

    Writeable returned arrays (since version 1.10.0):

    >>> a = np.zeros((3, 3))
    >>> np.einsum('ii->i', a)[:] = 1
    >>> a
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    Example of ellipsis use:

    >>> a = np.arange(6).reshape((3,2))
    >>> b = np.arange(12).reshape((4,3))
    >>> np.einsum('ki,jk->ij', a, b)
    array([[10, 28, 46, 64],
           [13, 40, 67, 94]])
    >>> np.einsum('ki,...k->i...', a, b)
    array([[10, 28, 46, 64],
           [13, 40, 67, 94]])
    >>> np.einsum('k...,jk', a, b)
    array([[10, 28, 46, 64],
           [13, 40, 67, 94]])

    Chained array operations. For more complicated contractions, speed ups
    might be achieved by repeatedly computing a 'greedy' path or pre-computing the
    'optimal' path and repeatedly applying it, using an
    `einsum_path` insertion (since version 1.12.0). Performance improvements can be
    particularly significant with larger arrays:

    >>> a = np.ones(64).reshape(2,4,8)

    Basic `einsum`: ~1520ms  (benchmarked on 3.1GHz Intel i5.)

    >>> for iteration in range(500):
    ...     _ = np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a)

    Sub-optimal `einsum` (due to repeated path calculation time): ~330ms

    >>> for iteration in range(500):
    ...     _ = np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a, optimize='optimal')

    Greedy `einsum` (faster optimal path approximation): ~160ms

    >>> for iteration in range(500):
    ...     _ = np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a, optimize='greedy')

    Optimal `einsum` (best usage pattern in some use cases): ~110ms

    >>> path = np.einsum_path('ijk,ilm,njm,nlk,abc->',a,a,a,a,a, optimize='optimal')[0]
    >>> for iteration in range(500):
    ...     _ = np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a, optimize=path)

    """
    # Special handling if out is specified
    specified_out = out is not None

    # If no optimization, run pure einsum
    if optimize is False:
        if specified_out:
            kwargs["out"] = out
        return c_einsum(*operands, **kwargs)

    # Check the kwargs to avoid a more cryptic error later, without having to
    # repeat default values here
    valid_einsum_kwargs = ["dtype", "order", "casting"]
    unknown_kwargs = [k for (k, v) in kwargs.items() if k not in valid_einsum_kwargs]
    if len(unknown_kwargs):
        raise TypeError("Did not understand the following kwargs: %s" % unknown_kwargs)

    # Build the contraction list and operand
    operands, contraction_list = einsum_path(
        *operands, optimize=optimize, einsum_call=True
    )

    # Handle order kwarg for output array, c_einsum allows mixed case
    output_order = kwargs.pop("order", "K")
    if output_order.upper() == "A":
        if all(arr.flags.f_contiguous for arr in operands):
            output_order = "F"
        else:
            output_order = "C"

    # Start contraction loop
    for num, contraction in enumerate(contraction_list):
        inds, idx_rm, einsum_str, remaining, blas = contraction
        tmp_operands = [operands.pop(x) for x in inds]

        # Do we need to deal with the output?
        handle_out = specified_out and ((num + 1) == len(contraction_list))

        # Call tensordot if still possible
        if blas:
            # Checks have already been handled
            input_str, results_index = einsum_str.split("->")
            input_left, input_right = input_str.split(",")

            tensor_result = input_left + input_right
            for s in idx_rm:
                tensor_result = tensor_result.replace(s, "")

            # Find indices to contract over
            left_pos, right_pos = [], []
            for s in sorted(idx_rm):
                left_pos.append(input_left.find(s))
                right_pos.append(input_right.find(s))

            # Contract!
            new_view = tensordot(
                *tmp_operands, axes=(tuple(left_pos), tuple(right_pos))
            )

            # Build a new view if needed
            if (tensor_result != results_index) or handle_out:
                if handle_out:
                    kwargs["out"] = out
                new_view = c_einsum(
                    tensor_result + "->" + results_index, new_view, **kwargs
                )

        # Call einsum
        else:
            # If out was specified
            if handle_out:
                kwargs["out"] = out

            # Do the contraction
            new_view = c_einsum(einsum_str, *tmp_operands, **kwargs)

        # Append new items and dereference what we can
        operands.append(new_view)
        del tmp_operands, new_view

    if specified_out:
        return out
    else:
        return asanyarray(operands[0], order=output_order)


def test_einsum():
    rng = np.random.RandomState(0)
    x = rng.randn(1, 2, 4, 8)
    y = rng.randn(1, 2, 4, 8)
    o = einsum("bthr,bThr->bhtT", x, y)
    print(x.shape, y.shape, "bthr,bThr->bhtT", o.shape)


# notes:
# "the true meaning of einsum ijk->ik is this: output[i,k] = sum(input[i,j,k] for j in range(input.shape[1]))"
# "ijk,ijk->ik is this: output[i,k] = sum(x[i,j,k] * y[i,j,k] for j in range(input.shape[1]))"
#
# np.einsum(lhs, [9, 1, 10, 8], rhs, [9, 5, 10, 8], [9, 10, 1, 5]).shape is (1, 4, 2, 2)
# basically lhs_axis_ids=bthr, rhs_axis_ids=bThr, out_axis_ids=bhtT
# but with the letters converted to arbitrary integers
#
# For a full breakdown of everything you can do with einsum, see numpy's source code:
# https://github.com/numpy/numpy/blob/3712815c3209c4769ccad60de110f2bd8f3763ec/numpy/core/src/multiarray/einsum.c.src#L728-L749
