import numpy as np

import jax
import jax.numpy as jnp
from jax import test_util as jtu
from jax.util import safe_map, safe_zip

from jax.experimental import djax
from jax.experimental.djax import (
    bbarray,
    ones_like,
    sin,
    add,
    iota,
    nonzero,
    reduce_sum,
    broadcast,
    dot,
    dot_general,
)

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip


from jax import core, lax
from jax._src.numpy import lax_numpy
import opt_einsum
import collections

from typing import Any, Sequence, FrozenSet, Optional, Tuple, Union, cast


def _einsum_contract_path(*operands, **kwargs):
    """Like opt_einsum.contract_path, with support for DimPolynomial shapes.

    We use opt_einsum.contract_path to compute the schedule, using a fixed
    constant for all dimension variables. This is safe because we throw an
    error if there are more than 1 contractions. Essentially, we just use
    opt_einsum.contract_path to parse the specification.
    """

    # Replace the polymorphic shapes with some concrete shapes for calling
    # into opt_einsum.contract_path, because the latter wants to compute the
    # sizes of operands and intermediate results.
    fake_ops = []
    for operand in operands:
        # We replace only array operands
        if not hasattr(operand, "dtype"):
            fake_ops.append(operand)
        else:
            shape = np.shape(operand)

            def fake_dim(d):
                # if core.is_constant_dim(d):
                #     return d
                # else:
                #     if not isinstance(d, Poly):
                #         raise TypeError(f"Encountered unexpected shape dimension {d}")
                #     # It is Ok to replace all polynomials with the same value. We may miss
                #     # here some errors due to non-equal dimensions, but we catch them
                #     # later.
                #     return 8
                return 8

            fake_ops.append(
                jax.ShapeDtypeStruct(tuple(map(fake_dim, shape)), operand.dtype)
            )

    contract_fake_ops, contractions = opt_einsum.contract_path(*fake_ops, **kwargs)
    if len(contractions) > 1:
        msg = (
            "Shape polymorphism is not yet supported for einsum with more than "
            f"one contraction {contractions}"
        )
        raise ValueError(msg)
    contract_operands = []
    for operand in contract_fake_ops:
        idx = tuple(i for i, fake_op in enumerate(fake_ops) if operand is fake_op)
        assert len(idx) == 1
        contract_operands.append(operands[idx[0]])
    return contract_operands, contractions


def djax_einsum(
    *operands, out=None, optimize="greedy", precision=None, _use_xeinsum=False
):
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

    if False:
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
            einsum_contract_path_fn = (
                lax_numpy._polymorphic_einsum_contract_path_handlers[
                    next(iter(non_constant_dim_types))
                ]
            )
    else:
        # einsum_contract_path_fn = opt_einsum.contract_path
        einsum_contract_path_fn = _einsum_contract_path
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
    operands = list(lax_numpy._promote_dtypes(*operands))

    def sum(x, axes):
        return lax.reduce(
            x,
            np.array(0, x.dtype),
            lax.add if x.dtype != lax_numpy.bool_ else lax.bitwise_or,
            axes,
        )

    def sum_uniques(operand, names, uniques):
        if uniques:
            axes = [names.index(name) for name in uniques]
            operand = sum(operand, axes)
            names = lax_numpy._removechars(names, uniques)
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
        s = lax_numpy.shape(operand)
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
        return lax_numpy.reshape(operand, tuple(new_shape)), "".join(new_names)

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

            # # handle cases where one side of a contracting or batch dimension is 1
            # # but its counterpart is not.
            # lhs, lhs_names = filter_singleton_dims(
            #     lhs, lhs_names, lax_numpy.shape(rhs), rhs_names
            # )
            # rhs, rhs_names = filter_singleton_dims(
            #     rhs, rhs_names, lax_numpy.shape(lhs), lhs_names
            # )

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

            lhs_batch, rhs_batch = lax_numpy.unzip2(
                (lhs_names.find(n), rhs_names.find(n)) for n in batch_names
            )

            # # NOTE(mattjj): this can fail non-deterministically in python3, maybe
            # # due to opt_einsum
            # assert lax_numpy._all(
            #     name in lhs_names
            #     and name in rhs_names
            #     and lhs.shape[lhs_names.index(name)] == rhs.shape[rhs_names.index(name)]
            #     for name in contracted_names
            # )

            # contract using lax.dot_general
            batch_names_str = "".join(batch_names)
            lhs_cont, rhs_cont = lax_numpy.unzip2(
                (lhs_names.index(n), rhs_names.index(n)) for n in contracted_names
            )
            deleted_names = batch_names_str + "".join(contracted_names)
            remaining_lhs_names = lax_numpy._removechars(lhs_names, deleted_names)
            remaining_rhs_names = lax_numpy._removechars(rhs_names, deleted_names)
            # Try both orders of lhs and rhs, in the hope that one of them means we
            # don't need an explicit transpose. opt_einsum likes to contract from
            # right to left, so we expect (rhs,lhs) to have the best chance of not
            # needing a transpose.
            names = batch_names_str + remaining_rhs_names + remaining_lhs_names
            if names == result_names:
                dimension_numbers = ((rhs_cont, lhs_cont), (rhs_batch, lhs_batch))
                # operand = lax.dot_general(rhs, lhs, dimension_numbers, precision)
                operand = dot_general(rhs, lhs, *dimension_numbers)
            else:
                names = batch_names_str + remaining_lhs_names + remaining_rhs_names
                dimension_numbers = ((lhs_cont, rhs_cont), (lhs_batch, rhs_batch))
                # operand = lax.dot_general(lhs, rhs, dimension_numbers, precision)
                operand = dot_general(lhs, rhs, *dimension_numbers)
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


def test1(self):
    @djax.djit
    def f(x, y):
        o = djax_einsum("bthr,bThr->bhtT", x, y)
        return o

    rng = np.random.RandomState(0)
    x = jnp.array(rng.randn(1, 2, 4, 8))
    y = jnp.array(rng.randn(1, 2, 4, 8))
    x = bbarray((1, 2, 4, 8), x)
    y = bbarray((1, 2, 4, 8), y)
    o = f(x, y)
    print(x.shape, y.shape, "bthr,bThr->bhtT", o.shape)
    return f, x, y


def test2(self):
    @djax.djit
    def f(x):
        nonzero_idx = nonzero(x)
        return reduce_sum(nonzero_idx)

    x = jnp.array([0, 1, 0, 1, 0, 1])
    ans = f(x)
    expected = np.sum(np.nonzero(x)[0])
    return ans


def test_linearize(self):
    @djax.djit
    def f(x):
        y = sin(x)
        return reduce_sum(y, axes=(0,))

    x = bbarray((5,), jnp.arange(2.0))
    with jax.enable_checks(False):  # TODO implement dxla_call abs eval rule
        z, f_lin = jax.linearize(f, x)
    z_dot = f_lin(ones_like(x))

    def g(x):
        return jnp.sin(x).sum()

    expected_z, expected_z_dot = jax.jvp(g, (np.arange(2.0),), (np.ones(2),))

    self.assertAllClose(np.array(z), expected_z, check_dtypes=False)
    self.assertAllClose(np.array(z_dot), expected_z_dot, check_dtypes=False)


import jax.test_util as tu


def test_djax():
    self = tu.JaxTestCase()
    test1(self)
    test2(self)
    test_linearize(self)
    print("")


if __name__ == "__main__":
    test_djax()
