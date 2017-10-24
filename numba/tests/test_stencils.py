#
# Copyright (c) 2017 Intel Corporation
# SPDX-License-Identifier: BSD-2-Clause
#

from __future__ import print_function, division, absolute_import

import sys
import numpy as np
import ast
import inspect
import operator
from dis import dis
import types as pytypes
import itertools
from contextlib import contextmanager
from copy import deepcopy

import numba
from numba import unittest_support as unittest
from numba import njit, stencil, types
from numba.compiler import compile_extra, Flags
from numba.targets import registry
from .support import tag
from numba.errors import LoweringError, TypingError


# for decorating tests, marking that Windows with Python 2.7 is not supported
_windows_py27 = (sys.platform.startswith('win32') and
                 sys.version_info[:2] == (2, 7))
_32bit = sys.maxsize <= 2 ** 32
_reason = 'parfors not supported'
skip_unsupported = unittest.skipIf(_32bit or _windows_py27, _reason)


@stencil
def stencil1_kernel(a):
    return 0.25 * (a[0, 1] + a[1, 0] + a[0, -1] + a[-1, 0])


@stencil(neighborhood=((-5, 0), ))
def stencil2_kernel(a):
    cum = a[-5]
    for i in range(-4, 1):
        cum += a[i]
    return 0.3 * cum


@stencil(cval=1.0)
def stencil3_kernel(a):
    return 0.25 * a[-2, 2]


@stencil
def stencil_multiple_input_kernel(a, b):
    return 0.25 * (a[0, 1] + a[1, 0] + a[0, -1] + a[-1, 0] +
                   b[0, 1] + b[1, 0] + b[0, -1] + b[-1, 0])


@stencil
def stencil_multiple_input_kernel_var(a, b, w):
    return w * (a[0, 1] + a[1, 0] + a[0, -1] + a[-1, 0] +
                b[0, 1] + b[1, 0] + b[0, -1] + b[-1, 0])


@stencil(standard_indexing=("b",))
def stencil_with_standard_indexing_1d(a, b):
    return a[-1] * b[0] + a[0] * b[1]


@stencil(standard_indexing=("b",))
def stencil_with_standard_indexing_2d(a, b):
    return a[0, 1] * b[0, 1] + a[1, 0] * b[1,
                                           0] + a[0, -1] * b[0, -1] + a[-1, 0] * b[-1, 0]


class TestStencilBase(unittest.TestCase):

    def __init__(self, *args):
        # flags for njit()
        self.cflags = Flags()
        self.cflags.set('nrt')

        # flags for njit(parallel=True)
        self.pflags = Flags()
        self.pflags.set('auto_parallel')
        self.pflags.set('nrt')
        super(TestStencilBase, self).__init__(*args)

    def _compile_this(self, func, sig, flags):
        return compile_extra(registry.cpu_target.typing_context,
                             registry.cpu_target.target_context, func, sig, None,
                             flags, {})

    def compile_parallel(self, func, sig):
        return self._compile_this(func, sig, flags=self.pflags)

    def compile_njit(self, func, sig):
        return self._compile_this(func, sig, flags=self.cflags)

    def compile_all(self, pyfunc, *args, **kwargs):
        sig = tuple([numba.typeof(x) for x in args])
        # compile with parallel=True
        cpfunc = self.compile_parallel(pyfunc, sig)
        # compile a standard njit of the original function
        cfunc = self.compile_njit(pyfunc, sig)
        return cfunc, cpfunc

    def check(self, no_stencil_func, pyfunc, *args):
        cfunc, cpfunc = self.compile_all(pyfunc, *args)
        # results without stencil macro
        expected = no_stencil_func(*args)
        # python result
        py_output = pyfunc(*args)

        # njit result
        njit_output = cfunc.entry_point(*args)

        # parfor result
        parfor_output = cpfunc.entry_point(*args)

        np.testing.assert_almost_equal(py_output, expected, decimal=1)
        np.testing.assert_almost_equal(njit_output, expected, decimal=1)
        np.testing.assert_almost_equal(parfor_output, expected, decimal=1)

        # make sure parfor set up scheduling
        self.assertIn('@do_scheduling', cpfunc.library.get_llvm_str())


class TestStencil(TestStencilBase):

    def __init__(self, *args, **kwargs):
        super(TestStencilBase, self).__init__(*args, **kwargs)

    @skip_unsupported
    @tag('important')
    def test_stencil1(self):
        """Tests whether the optional out argument to stencil calls works.
        """
        def test_with_out(n):
            A = np.arange(n**2).reshape((n, n))
            B = np.zeros(n**2).reshape((n, n))
            B = stencil1_kernel(A, out=B)
            return B

        def test_without_out(n):
            A = np.arange(n**2).reshape((n, n))
            B = stencil1_kernel(A)
            return B

        def test_impl_seq(n):
            A = np.arange(n**2).reshape((n, n))
            B = np.zeros(n**2).reshape((n, n))
            for i in range(1, n - 1):
                for j in range(1, n - 1):
                    B[i, j] = 0.25 * (A[i, j + 1] + A[i + 1,
                                                      j] + A[i, j - 1] + A[i - 1, j])
            return B

        n = 100
        self.check(test_impl_seq, test_with_out, n)
        self.check(test_impl_seq, test_without_out, n)

    @skip_unsupported
    @tag('important')
    def test_stencil2(self):
        """Tests whether the optional neighborhood argument to the stencil
        decorate works.
        """
        def test_seq(n):
            A = np.arange(n)
            B = stencil2_kernel(A)
            return B

        def test_impl_seq(n):
            A = np.arange(n)
            B = np.zeros(n)
            for i in range(5, len(A)):
                B[i] = 0.3 * sum(A[i - 5:i + 1])
            return B

        n = 100
        self.check(test_impl_seq, test_seq, n)
        # variable length neighborhood in numba.stencil call
        # only supported in parallel path

        def test_seq(n, w):
            A = np.arange(n)

            def stencil2_kernel(a, w):
                cum = a[-w]
                for i in range(-w + 1, w + 1):
                    cum += a[i]
                return 0.3 * cum
            B = numba.stencil(stencil2_kernel, neighborhood=((-w, w), ))(A, w)
            return B

        def test_impl_seq(n, w):
            A = np.arange(n)
            B = np.zeros(n)
            for i in range(w, len(A) - w):
                B[i] = 0.3 * sum(A[i - w:i + w + 1])
            return B
        n = 100
        w = 5
        cpfunc = self.compile_parallel(test_seq, (types.intp, types.intp))
        expected = test_impl_seq(n, w)
        # parfor result
        parfor_output = cpfunc.entry_point(n, w)
        np.testing.assert_almost_equal(parfor_output, expected, decimal=1)
        self.assertIn('@do_scheduling', cpfunc.library.get_llvm_str())
        # test index_offsets

        def test_seq(n, w, offset):
            A = np.arange(n)

            def stencil2_kernel(a, w):
                cum = a[-w + 1]
                for i in range(-w + 1, w + 1):
                    cum += a[i + 1]
                return 0.3 * cum
            B = numba.stencil(stencil2_kernel, neighborhood=((-w, w), ),
                              index_offsets=(-offset, ))(A, w)
            return B

        offset = 1
        cpfunc = self.compile_parallel(test_seq, (types.intp, types.intp,
                                                  types.intp))
        parfor_output = cpfunc.entry_point(n, w, offset)
        np.testing.assert_almost_equal(parfor_output, expected, decimal=1)
        self.assertIn('@do_scheduling', cpfunc.library.get_llvm_str())
        # test slice in kernel

        def test_seq(n, w, offset):
            A = np.arange(n)

            def stencil2_kernel(a, w):
                return 0.3 * np.sum(a[-w + 1:w + 2])
            B = numba.stencil(stencil2_kernel, neighborhood=((-w, w), ),
                              index_offsets=(-offset, ))(A, w)
            return B

        offset = 1
        cpfunc = self.compile_parallel(test_seq, (types.intp, types.intp,
                                                  types.intp))
        parfor_output = cpfunc.entry_point(n, w, offset)
        np.testing.assert_almost_equal(parfor_output, expected, decimal=1)
        self.assertIn('@do_scheduling', cpfunc.library.get_llvm_str())

    @skip_unsupported
    @tag('important')
    def test_stencil3(self):
        """Tests whether a non-zero optional cval argument to the stencil
        decorator works.  Also tests integer result type.
        """
        def test_seq(n):
            A = np.arange(n**2).reshape((n, n))
            B = stencil3_kernel(A)
            return B

        test_njit = njit(test_seq)
        test_par = njit(test_seq, parallel=True)

        n = 5
        seq_res = test_seq(n)
        njit_res = test_njit(n)
        par_res = test_par(n)

        self.assertTrue(seq_res[0, 0] == 1.0 and seq_res[4, 4] == 1.0)
        self.assertTrue(njit_res[0, 0] == 1.0 and njit_res[4, 4] == 1.0)
        self.assertTrue(par_res[0, 0] == 1.0 and par_res[4, 4] == 1.0)

    @skip_unsupported
    @tag('important')
    def test_stencil_standard_indexing_1d(self):
        """Tests standard indexing with a 1d array.
        """
        def test_seq(n):
            A = np.arange(n)
            B = [3.0, 7.0]
            C = stencil_with_standard_indexing_1d(A, B)
            return C

        def test_impl_seq(n):
            A = np.arange(n)
            B = [3.0, 7.0]
            C = np.zeros(n)

            for i in range(1, n):
                C[i] = A[i - 1] * B[0] + A[i] * B[1]
            return C

        n = 100
        self.check(test_impl_seq, test_seq, n)

    @skip_unsupported
    @tag('important')
    def test_stencil_standard_indexing_2d(self):
        """Tests standard indexing with a 2d array.
        """
        def test_seq(n):
            A = np.arange(n**2).reshape((n, n))
            B = np.ones((3, 3))
            C = stencil_with_standard_indexing_2d(A, B)
            return C

        def test_impl_seq(n):
            A = np.arange(n**2).reshape((n, n))
            B = np.ones((3, 3))
            C = np.zeros(n**2).reshape((n, n))

            for i in range(1, n - 1):
                for j in range(1, n - 1):
                    C[i, j] = (A[i, j + 1] * B[0, 1] + A[i + 1, j] * B[1, 0] +
                               A[i, j - 1] * B[0, -1] + A[i - 1, j] * B[-1, 0])
            return C

        n = 5
        self.check(test_impl_seq, test_seq, n)

    @skip_unsupported
    @tag('important')
    def test_stencil_multiple_inputs(self):
        """Tests whether multiple inputs of the same size work.
        """
        def test_seq(n):
            A = np.arange(n**2).reshape((n, n))
            B = np.arange(n**2).reshape((n, n))
            C = stencil_multiple_input_kernel(A, B)
            return C

        def test_impl_seq(n):
            A = np.arange(n**2).reshape((n, n))
            B = np.arange(n**2).reshape((n, n))
            C = np.zeros(n**2).reshape((n, n))
            for i in range(1, n - 1):
                for j in range(1, n - 1):
                    C[i,
                      j] = 0.25 * (A[i,
                                     j + 1] + A[i + 1,
                                                j] + A[i,
                                                       j - 1] + A[i - 1,
                                                                  j] + B[i,
                                                                         j + 1] + B[i + 1,
                                                                                    j] + B[i,
                                                                                           j - 1] + B[i - 1,
                                                                                                      j])
            return C

        n = 3
        self.check(test_impl_seq, test_seq, n)
        # test stencil with a non-array input

        def test_seq(n):
            A = np.arange(n**2).reshape((n, n))
            B = np.arange(n**2).reshape((n, n))
            w = 0.25
            C = stencil_multiple_input_kernel_var(A, B, w)
            return C
        self.check(test_impl_seq, test_seq, n)

    @skip_unsupported
    @tag('important')
    def test_stencil_call(self):
        """Tests 2D numba.stencil calls.
        """
        def test_impl1(n):
            A = np.arange(n**2).reshape((n, n))
            B = np.zeros(n**2).reshape((n, n))
            numba.stencil(lambda a: 0.25 * (a[0, 1] + a[1, 0] + a[0, -1]
                                            + a[-1, 0]))(A, out=B)
            return B

        def test_impl2(n):
            A = np.arange(n**2).reshape((n, n))
            B = np.zeros(n**2).reshape((n, n))

            def sf(a):
                return 0.25 * (a[0, 1] + a[1, 0] + a[0, -1] + a[-1, 0])
            B = numba.stencil(sf)(A)
            return B

        def test_impl_seq(n):
            A = np.arange(n**2).reshape((n, n))
            B = np.zeros(n**2).reshape((n, n))
            for i in range(1, n - 1):
                for j in range(1, n - 1):
                    B[i, j] = 0.25 * (A[i, j + 1] + A[i + 1,
                                                      j] + A[i, j - 1] + A[i - 1, j])
            return B

        n = 100
        self.check(test_impl_seq, test_impl1, n)
        self.check(test_impl_seq, test_impl2, n)

    @skip_unsupported
    @tag('important')
    def test_stencil_call_1D(self):
        """Tests 1D numba.stencil calls.
        """
        def test_impl(n):
            A = np.arange(n)
            B = np.zeros(n)
            numba.stencil(lambda a: 0.3 * (a[-1] + a[0] + a[1]))(A, out=B)
            return B

        def test_impl_seq(n):
            A = np.arange(n)
            B = np.zeros(n)
            for i in range(1, n - 1):
                B[i] = 0.3 * (A[i - 1] + A[i] + A[i + 1])
            return B

        n = 100
        self.check(test_impl_seq, test_impl, n)


class pyStencilGenerator:
    """
    Holds the classes and methods needed to generate a python stencil implementation from a kernel purely using AST transforms.
    """

    class Builder:
        """
        Provides code generation for the AST manipulation pipeline
        """

        ids = 'ijklmno'  # 7 is enough for fortran
        # builder functions

        def genalloc_return(self, orig, var, dtype_var, init_val=0):
            new = ast.Assign(
                targets=[
                    ast.Name(
                        id=var,
                        ctx=ast.Store())],
                value=ast.BinOp(op=ast.Mult(), left=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(
                            id='np',
                            ctx=ast.Load()),
                        attr='ones',
                        ctx=ast.Load()),
                    args=[
                        ast.Attribute(
                            value=ast.Name(
                                id=orig,
                                ctx=ast.Load()),
                            attr='shape',
                            ctx=ast.Load()),
                            self.genCall('type', [dtype_var.id]).value],
                    keywords=[],
                    starargs=None,
                    kwargs=None), right=self.genNum(init_val)))
            return new

        def genassign(self, var, value, index_names):
            elts_info = [ast.Name(id=x, ctx=ast.Load()) for x in index_names]
            new = ast.Assign(
                targets=[
                    ast.Subscript(
                        value=ast.Name(
                            id=var,
                            ctx=ast.Load()),
                        slice=ast.Index(
                            value=ast.Tuple(
                                elts=elts_info,
                                ctx=ast.Load())),
                        ctx=ast.Store())],
                value=value)
            return new

        def genloop(self, var, start=0, stop=0, body=None):
            if isinstance(start, int):
                start_val = ast.Num(n=start)
            else:
                start_val = start
            if isinstance(stop, int):
                stop_val = ast.Num(n=stop)
            else:
                stop_val = stop
            return ast.For(
                target=ast.Name(id=var, ctx=ast.Store()),
                iter=ast.Call(
                    func=ast.Name(id='range', ctx=ast.Load()),
                    args=[start_val, stop_val],
                    keywords=[],
                    starargs=None, kwargs=None),
                body=body, orelse=[])

        def genreturn(self, var):
            return ast.Return(value=ast.Name(id=var, ctx=ast.Load()))

        def genslice(self, value):
            return ast.Index(value=ast.Num(n=value))

        def genattr(self, name, attr):
            return ast.Attribute(
                value=ast.Name(id=name, ctx=ast.Load()),
                attr=attr, ctx=ast.Load())

        def gensubscript(self, name, attr, index, offset=None):
            attribute = self.genattr(name, attr)
            slise = self.genslice(index)
            ss = ast.Subscript(value=attribute, slice=slise, ctx=ast.Load())
            if offset:
                pm = ast.Add() if offset >= 0 else ast.Sub()
                ss = ast.BinOp(left=ss, op=pm, right=ast.Num(n=abs(offset)))
            return ss

        def genNum(self, value):
            if abs(value) >= 0:
                return ast.Num(value)
            else:
                return ast.UnaryOp(ast.USub(), ast.Num(-value))

        def genCall(self, call_name, args):
            fixed_args = [ast.Name(id='%s' % x, ctx=ast.Load()) for x in args]
            return ast.Expr(value=ast.Call(
                            func=ast.Name(id=call_name, ctx=ast.Load()),
                            args=fixed_args, keywords=[]), ctx=ast.Load())

    # AST transformers
    class FoldConst(ast.NodeTransformer, Builder):
        """
        Folds const expr, this is so const expressions in the relidx are
        more easily handled
        """

        # just support a few for testing purposes
        supported_ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
        }

        def visit_BinOp(self, node):
            # does const expr folding
            node = self.generic_visit(node)

            op = self.supported_ops.get(node.op.__class__)
            lhs = getattr(node, 'left', None)
            rhs = getattr(node, 'right', None)

            if not (lhs and rhs and op):
                return node

            if (isinstance(lhs, ast.Num) and
                    isinstance(rhs, ast.Num)):
                return ast.Num(op(node.left.n, node.right.n))
            else:
                return node

    class FixRelIndex(ast.NodeTransformer, Builder):
        """ Fixes the relative indexes to be written in as induction index + relative index
        """

        def __init__(self, argnames, standard_indexing, *args, **kwargs):
            super(ast.NodeTransformer, self).__init__(*args, **kwargs)
            self.argnames = argnames
            self.idx_len = -1
            self.mins = None
            self.maxes = None
            self.imin = np.iinfo(int).min
            self.imax = np.iinfo(int).max
            self.standard_indexing = standard_indexing if standard_indexing else []

        def visit_Subscript(self, node):
            node = self.generic_visit(node)

            if (node.value.id in self.argnames) and (node.value.id not in self.standard_indexing):
                if isinstance(node.slice.value, ast.Tuple):
                    idx = []
                    for x, val in enumerate(node.slice.value.elts):
                        idx.append(
                            ast.BinOp(
                                left=ast.Name(
                                    id='__%s' %
                                    self.ids[x],
                                    ctx=ast.Load()),
                                op=ast.Add(),
                                right=val,
                                ctx=ast.Load()))
                    if self.idx_len == -1:
                        self.idx_len = len(idx)
                    else:
                        if(self.idx_len != len(idx)):
                            raise ValueError(
                                "Relative indexing mismatch detected")
                    if isinstance(node.ctx, ast.Store):
                        msg = "Assignments to array passed to stencil kernels is not allowed"
                        raise ValueError(msg)
                    context = ast.Load()
                    newnode = ast.Subscript(
                        value=node.value,
                        slice=ast.Index(
                            value=ast.Tuple(
                                elts=idx,
                                ctx=ast.Load()),
                            ctx=ast.Load()),
                        ctx=context)
                    ast.copy_location(newnode, node)
                    ast.fix_missing_locations(newnode)

                    # now work out max/min for index ranges i.e. stencil size
                    if self.mins is None and self.maxes is None:
                        # first pass
                        self.mins = [self.imax] * self.idx_len
                        self.maxes = [self.imin] * self.idx_len

                    def getval(node):
                        if isinstance(node, ast.Num):
                            return node.n
                        elif isinstance(node, ast.UnaryOp):
                            return -node.operand.n
                        else:
                            raise ValueError("Unknown indexing operation")

                    for x, lnode in enumerate(node.slice.value.elts):
                        relvalue = getval(lnode)
                        if relvalue < self.mins[x]:
                            self.mins[x] = relvalue
                        if relvalue > self.maxes[x]:
                            self.maxes[x] = relvalue
                    return newnode
            else:
                return node

        def get_idx_len(self):
            if self.idx_len == -1:
                raise ValueError(
                    'Transform has not been run/no indexes found')
            else:
                return self.idx_len

    class FixFunc(ast.NodeTransformer, Builder):
        """ The main function rewriter, takes the body of the kernel and generates:
         * checking function calls
         * return value allocation
         * loop nests
         * return site
         * Function definition as an entry point
        """

        def __init__(self, original_kernel, loop_data, argnames, retty, cval, standard_indexing, *args, **kwargs):
            super(ast.NodeTransformer, self).__init__(*args, **kwargs)
            self.original_kernel = original_kernel
            self.loop_data = loop_data
            self.argnames = argnames
            self.retty = retty
            self.standard_indexing = standard_indexing if standard_indexing else []
            self.relidx_args = [x for x in self.argnames if x not in self.standard_indexing]
            # switch cval to python type
            if hasattr(cval, 'dtype'):
                self.cval = cval.tolist()
            else:
                self.cval = cval
            self.stencil_arr = argnames[0]

        def visit_Return(self, node):
            return node.value

        def visit_FunctionDef(self, node):
            # make sure there is a return and its the last statement
            assert(isinstance(node.body[-1], ast.Return))

            # this function validates arguments and is injected into the top
            # of the stencil call
            def check_stencil_arrays(*args):
                # the first has to be an array due to parfors requirements
                init_shape = args[0].shape
                for x in args[1:]:
                    if hasattr(x, 'shape'):
                        if init_shape != x.shape:
                            raise ValueError("Input stencil arrays do not commute")

            checksrc = inspect.getsource(check_stencil_arrays)
            check_impl = ast.parse(
                checksrc.strip()).body[0]  # don't need module
            ast.fix_missing_locations(check_impl)

            checker_call = self.genCall('check_stencil_arrays', self.relidx_args)           

            blk = []
            for b in node.body:
                # strip return block
                if isinstance(b, ast.Return):
                    blk.append(b.value)
                else:
                    blk.append(b)
            retvar = '__b%s' % np.random.randint(100)
            nloops = self.loop_data['number']
            # replay blocks, switch return for assign
            loop_body = [
                *blk[: -1],
                self.genassign(
                    retvar, blk[-1],
                    ['__%s' % self.ids[l] for l in range(nloops)])]
            nloops -= 1  # account for inner loop peeled/0 indexing

            def computebound(mins, maxs):
                if mins == maxs:
                    minlim = 0 if mins >= 0 else -mins
                    maxlim = -maxs if maxs > 0 else 0
                else:
                    minlim = 0 if mins >= 0 else -mins  # ? -mins?
                    maxlim = -maxs if maxs > 0 else 0
                return (minlim, maxlim)

            minlim, maxlim = computebound(
                self.loop_data['mins'][-1],
                self.loop_data['maxes'][-1])

            minbound = minlim
            maxbound = self.gensubscript(
                self.stencil_arr, 'shape', nloops, maxlim)

            # this is the inner loop
            loops = self.genloop(
                '__%s' %
                self.ids[nloops],
                minbound,
                maxbound,
                body=loop_body)

            # tile outer loops
            for l in range(nloops):
                minlim, maxlim = computebound(
                    self.loop_data['mins'][nloops - l - 1],
                    self.loop_data['maxes'][nloops - l - 1])
                minbound = minlim
                maxbound = self.gensubscript(
                    self.stencil_arr, 'shape', nloops - l - 1, maxlim)
                loops = self.genloop(
                    '__%s' % self.ids[nloops - l - 1],
                    minbound, maxbound, body=[loops])
            ast.copy_location(loops, node)
            rettyname = self.retty.targets[0]
            allocate = self.genalloc_return(
                self.stencil_arr, retvar, rettyname, self.cval)
            ast.copy_location(allocate, node)
            returner = self.genreturn(retvar)
            ast.copy_location(returner, node)
            new = ast.FunctionDef(
                name='__%s' %
                node.name,
                args=node.args,
                body=[
                    check_impl,
                    checker_call,
                    self.original_kernel,
                    self.retty,
                    allocate,
                    loops,
                    returner],
                decorator_list=[])
            ast.copy_location(new, node)
            return new

    class GetArgNames(ast.NodeVisitor, Builder):
        """ Gets the argument names """

        def __init__(self, *args, **kwargs):
            super(ast.NodeVisitor, self).__init__(*args, **kwargs)
            self._argnames = None
            self._kwargnames = None
            self.retty = None
            self.original_kernel = None
            
        def visit_FunctionDef(self, node):
            if self._argnames is not None or self._kwargnames is not None:
                raise RuntimeError("multiple definition of function/args?")
            self._argnames = [x.arg for x in node.args.args]
            if node.args.kwarg:
                self._kwargnames = [x.arg for x in node.args.kwarg]
            compute_retdtype = self.genCall(node.name, self._argnames)
            self.retty = ast.Assign(targets=[ast.Name(
            id='__retdtype',
            ctx=ast.Store())], value=compute_retdtype.value)
            self.original_kernel = ast.fix_missing_locations(deepcopy(node))
            self.generic_visit(node)
            

        def get_arg_names(self):
            return self._argnames

    class FixCalls(ast.NodeTransformer):
        """ Fixes call sites for astor (in case it is in use) """

        def visit_Call(self, node):
            self.generic_visit(node)
            # astor needs starargs and kwargs
            new = ast.Call(
                node.func,
                node.args,
                node.keywords,
                starargs=None,
                kwargs=None)
            return new

    def generate_stencil_tree(self, func, cval, standard_indexing):
        """
        Generates the AST tree for a stencil based on func and cval.
        """
        src = inspect.getsource(func)
        tree = ast.parse(src.strip())

        DEBUG = True
        DEBUG = False
        if DEBUG:
            print("ORIGINAL")
            print(ast.dump(tree))

        def pipeline(tree):
            """ the pipeline of manipulations """
            # get the arg names
            argnamegetter = self.GetArgNames()
            argnamegetter.visit(tree)
            argnm = argnamegetter.get_arg_names()
            if standard_indexing:
                for x in standard_indexing:
                    if x not in argnm:
                        raise ValueError("Non-existent variable specified in standard_indexing")

            # return type computation
            # gets an ast to generate a return type
            retty = argnamegetter.retty
            original_kernel = argnamegetter.original_kernel

            # fold consts
            fold_const = self.FoldConst()
            fold_const.visit(tree)

            # rewrite the relative indices as induced indices
            relidx_fixer = self.FixRelIndex(argnm, standard_indexing)
            relidx_fixer.visit(tree)
            index_len = relidx_fixer.get_idx_len()

            # generate the function body and loop nests and assemble
            fixer = self.FixFunc(original_kernel,
                {'number': index_len, 'maxes': relidx_fixer.maxes,
                 'mins': relidx_fixer.mins},
                argnm, retty, cval, standard_indexing)
            fixer.visit(tree)

            # fix up the call sites so they work better with astor
            callFixer = self.FixCalls()
            callFixer.visit(tree)
            ast.fix_missing_locations(tree.body[0])

        # run the pipeline of transforms on the tree
        pipeline(tree)

        if DEBUG:
            print("\n\n\nNEW")
            print(ast.dump(tree, include_attributes=True))
            try:
                import astor
                print(astor.to_source(tree))
            except Exception:
                pass

        return tree


def pyStencil(func_or_mode='constant', **options):
    """
    A pure python implementation of stencil functionality, equivalent to StencilFunc
    """

    if not isinstance(func_or_mode, str):
        mode = 'constant'  # default style
        func = func_or_mode
    else:
        assert isinstance(func_or_mode, str), """stencil mode should be
                                                        a string"""
        mode = func_or_mode
        func = None

    for option in options:
        if option not in ["cval", "standard_indexing", "neighborhood"]:
            raise ValueError("Unknown stencil option " + option)

    if mode != 'constant':
        raise ValueError("Unsupported mode style " + mode)

    cval = options.get('cval', 0)
    standard_indexing = options.get('standard_indexing', None)

    # generate a new AST tree from the kernel func
    tree = pyStencilGenerator().generate_stencil_tree(func, cval, standard_indexing)

    # breathe life into the tree
    mod_code = compile(tree, filename="<ast>", mode="exec")
    func_code = mod_code.co_consts[0]
    full_func = pytypes.FunctionType(func_code, globals())

    return full_func


class TestManyStencils(TestStencilBase):

    def check(self, pyfunc, *args, options={}, expected_exception=None, **kwargs):
        """
        For a given kernel:

        The expected result is computed from a pyStencil version of the stencil.

        The following results are then computed:
        * from a pure @stencil decoration of the kernel
        * from the njit of a trivial wrapper function around the pure @stencil decorated function
        * from the njit(parallel=True) of a trivial wrapper function around the pure @stencil decorated function

        The results are then compared.
        """

        # DEBUG print output arrays
        DEBUG_OUTPUT = True
        DEBUG_OUTPUT = False
        
        # collect fails
        should_fail=[]
        should_not_fail=[]
        
        # runner that handles fails
        @contextmanager
        def errorhandler(exty = None, usecase=None):
            try:
                yield
            except Exception as e:
                if exty is not None:
                    lexty = exty if hasattr(exty, '__iter__') else [exty,]
                    found = False
                    for ex in lexty:
                        found |= isinstance(e, ex)
                    if not found:
                        raise
                else:
                    should_not_fail.append((usecase, e))
            else:
                if exty is not None:
                    should_fail.append(usecase)

        if isinstance(expected_exception, dict):
            pystencil_ex = expected_exception['pyStencil']
            stencil_ex = expected_exception['stencil']
            njit_ex = expected_exception['njit']
            parfor_ex = expected_exception['parfor']
        else:
            pystencil_ex = expected_exception
            stencil_ex = expected_exception
            njit_ex = expected_exception
            parfor_ex = expected_exception
            

        stencil_args = {'func_or_mode': pyfunc, **options}

        expected_present = True
        # try running the pyStencil version, if it raises then run the
        # 
        try: 
            # ast impl
            ast_impl = pyStencil(func_or_mode=pyfunc, **options)
            expected = ast_impl(*args)
            if DEBUG_OUTPUT:
                print("\nExpected:\n", expected)
        except Exception as ex:
            # check exception is expected
            with errorhandler(pystencil_ex, "pyStencil"):
                raise ex
            pyStencil_unhandled_ex = ex
            expected_present = False

        stencilfunc_output = None
        with errorhandler(stencil_ex, "@stencil"):
            stencil_func_impl = stencil(**stencil_args)
            # stencil result
            stencilfunc_output = stencil_func_impl(*args)

        # wrapped stencil impl, could this be generated?
        if len(args) == 1:
            def wrap_stencil(arg0):
                return stencil_func_impl(arg0)
        elif len(args) == 2:
            def wrap_stencil(arg0, arg1):
                return stencil_func_impl(arg0, arg1)
        elif len(args) == 3:
            def wrap_stencil(arg0, arg1, arg2):
                return stencil_func_impl(arg0, arg1, arg2)
        else:
            raise ValueError(
                "Up to 3 arguments can be provided, found %s" %
                len(args))


        sig = tuple([numba.typeof(x) for x in args])
        
        njit_output = None
        with errorhandler(njit_ex, "njit"):
            wrapped_cfunc = self.compile_njit(wrap_stencil, sig)
            # njit result
            njit_output = wrapped_cfunc.entry_point(*args)
            
        parfor_output = None
        with errorhandler(parfor_ex, "parfors"):
            wrapped_cpfunc = self.compile_parallel(wrap_stencil, sig)
            # parfor result
            parfor_output = wrapped_cpfunc.entry_point(*args)
            
        if DEBUG_OUTPUT:
            print("\n@stencil_output:\n", stencilfunc_output)
            print("\nnjit_output:\n", njit_output)
            print("\nparfor_output:\n", parfor_output)


        if expected_present:
            try:
                if not stencil_ex:
                    np.testing.assert_almost_equal(
                        stencilfunc_output, expected, decimal=1)
                    self.assertEqual(expected.dtype, stencilfunc_output.dtype)
            except Exception as e:
                should_not_fail.append(('@stencil', e))
                print("@stencil failed: %s" % str(e))

            try:
                if not njit_ex:
                    np.testing.assert_almost_equal(
                        njit_output, expected, decimal=1)
                    self.assertEqual(expected.dtype, njit_output.dtype)
            except Exception as e:
                should_not_fail.append(('njit', e))
                print("@njit failed: %s" % str(e))

            try:
                if not parfor_ex:
                    np.testing.assert_almost_equal(
                        parfor_output, expected, decimal=1)
                    self.assertEqual(expected.dtype, parfor_output.dtype)
                    try:
                        self.assertIn(
                            '@do_scheduling',
                            wrapped_cpfunc.library.get_llvm_str())
                    except AssertionError:
                        msg = 'Could not find `@do_scheduling` in LLVM IR'
                        raise AssertionError(msg)
            except Exception as e:
                should_not_fail.append(('parfors', e))
                print("@njit(parallel=True) failed: %s" % str(e))


        if DEBUG_OUTPUT:
            print("\n\n")
            
        if should_fail:
            msg = ["%s" % x for x in should_fail]
            raise RuntimeError("The following implementations should have raised an exception but did not:\n%s" % msg )

        if should_not_fail:
            impls = ["%s" % x[0] for x in should_not_fail]
            errs = ''.join(["%s: Message: %s\n\n" % x for x in should_not_fail])
            str1 = "The following implementations should not have raised an exception but did:\n%s\n" % impls
            str2 = "Errors were:\n\n %s " % errs
            raise RuntimeError( str1 + str2 )

        if not expected_present:
            if expected_exception is None:
                raise RuntimeError("pyStencil failed, was not caught/expected", pyStencil_unhandled_ex)
                

    def exception_dict(self, **kwargs):
        d = dict()
        d['pyStencil'] = None
        d['stencil'] = None
        d['njit'] = None
        d['parfor'] = None
        for k, v in kwargs.items():
            d[k] = v
        return d

    @skip_unsupported
    @tag('important')
    # This should work but gives:
    # TypingError, array(float64, 2d, C) returning into array(int64, 2d, C)
    def test_basic00(self):
        """rel index"""
        def kernel(a):
            return a[0, 0]
        a = np.arange(12).reshape(3, 4)
        self.check(kernel, a)

    def test_basic01(self):
        """rel index add const"""
        def kernel(a):
            return a[0, 1]
        a = np.arange(12.).reshape(3, 4)
        self.check(kernel, a)

    def test_basic02(self):
        """rel index add const"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[0, -1]
        self.check(kernel, a)

    def test_basic03(self):
        """rel index add const"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[1, 0]
        self.check(kernel, a)

    def test_basic04(self):
        """rel index add const"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[-1, 0]
        self.check(kernel, a)

    def test_basic05(self):
        """rel index add const"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[-1, 1]
        self.check(kernel, a)

    def test_basic06(self):
        """rel index add const"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[1, -1]
        self.check(kernel, a)

    def test_basic07(self):
        """rel index add const"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[1, 1]
        self.check(kernel, a)

    def test_basic08(self):
        """rel index add const"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[-1, -1]
        self.check(kernel, a)

    def test_basic09(self):
        """rel index add const"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[-2, 2]
        self.check(kernel, a)

    def test_basic10(self):
        """rel index add const"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[0, 0] + a[1, 0]
        self.check(kernel, a)

    def test_basic11(self):
        """rel index add const"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[-1, 0] + a[1, 0]
        self.check(kernel, a)

    def test_basic12(self):
        """rel index add const"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[-1, 1] + a[1, -1]
        self.check(kernel, a)

    def test_basic13(self):
        """rel index add const"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[-1, -1] + a[1, 1]
        self.check(kernel, a)

    def test_basic14(self):
        """rel index add domain change const"""
        a = np.arange(12).reshape(3, 4)

        def kernel(a):
            return a[0, 0] + 1j
        self.check(kernel, a)

    def test_basic14b(self):
        """rel index add domain change const"""
        a = np.arange(12).reshape(3, 4)

        def kernel(a):
            t = 1.j
            return a[0, 0] + t
        self.check(kernel, a)

    def test_basic15(self):
        """two rel index, add const"""
        a = np.arange(12).reshape(3, 4)

        def kernel(a):
            return a[0, 0] + a[1, 0] + 1.
        self.check(kernel, a)

    # FIXME: pyStencil fails, IndexError
    def test_basic16(self):
        """two rel index OOB, add const"""
        a = np.arange(12).reshape(3, 4)

        def kernel(a):
            return a[0, 0] + a[10, 0] + 1.
        self.check(kernel, a)

    def test_basic17(self):
        """two rel index boundary test, add const"""
        a = np.arange(12).reshape(3, 4)

        def kernel(a):
            return a[0, 0] + a[2, 0] + 1.
        self.check(kernel, a)

    def test_basic18(self):
        """two rel index boundary test, add const"""
        a = np.arange(12).reshape(3, 4)

        def kernel(a):
            return a[0, 0] + a[-2, 0] + 1.
        self.check(kernel, a)

    def test_basic19(self):
        """two rel index boundary test, add const"""
        a = np.arange(12).reshape(3, 4)

        def kernel(a):
            return a[0, 0] + a[0, 3] + 1.
        self.check(kernel, a)

    def test_basic20(self):
        """two rel index boundary test, add const"""
        a = np.arange(12).reshape(3, 4)

        def kernel(a):
            return a[0, 0] + a[0, -3] + 1.
        self.check(kernel, a)

    def test_basic21(self):
        """same rel, add const"""
        a = np.arange(12).reshape(3, 4)

        def kernel(a):
            return a[0, 0] + a[0, 0] + 1.
        self.check(kernel, a)

    def test_basic22(self):
        """rel idx const expr folding, add const"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[1 + 0, 0] + a[0, 0] + 1.
        self.check(kernel, a)

    def test_basic23(self):
        """rel idx, work in body"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            x = np.sin(10 + a[2, 1])
            return a[1 + 0, 0] + a[0, 0] + x
        self.check(kernel, a)

    def test_basic23a(self):
        """rel idx, dead code should not impact rel idx"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            x = np.sin(10 + a[2, 1])
            return a[1 + 0, 0] + a[0, 0]
        self.check(kernel, a)

    def test_basic24(self):
        """1d idx on 2d arr"""
        a = np.arange(12).reshape(3, 4)

        def kernel(a):
            return a[0] + 1.
        self.check(kernel, a, expected_exception=[ValueError, TypingError])

    # Should this work at all? does it make sense?
    # pyStencil raises as there's no rel indices
    def test_basic25(self):
        """1d idx on 2d arr"""
        a = np.arange(12).reshape(3, 4)

        def kernel(a):
            return 1.
        self.check(kernel, a, expected_exception=[ValueError])

    def test_basic26(self):
        """3d arr"""
        a = np.arange(64).reshape(4, 8, 2)

        def kernel(a):
            return a[0, 0, 0] - a[0, 1, 0] + 1.
        self.check(kernel, a)

    def test_basic27(self):
        """4d arr"""
        a = np.arange(128).reshape(4, 8, 2, 2)

        def kernel(a):
            return a[0, 0, 0, 0] - a[0, 1, 0, -1] + 1.
        self.check(kernel, a)

    def test_basic28(self):
        """type widen """
        a = np.arange(12).reshape(3, 4).astype(np.float32)

        def kernel(a):
            return a[0, 0] + np.float64(10.)
        self.check(kernel, a)

    def test_basic29(self):
        """const index from func """
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[0, int(np.cos(0))]
        self.check(kernel, a, expected_exception=[ValueError, LoweringError])

    def test_basic30(self):
        """signed zeros"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[-0, -0]
        self.check(kernel, a)

    # this passes parfors, fails in pyStencil
    def test_basic31(self):
        """does a const propagate?"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            t = 1
            return a[t, 0]
        self.check(kernel, a)
        
    # this fails all, need to decide whether to entirely support const prop + fold
    # or not.
    def test_basic31b(self):
        """does a const propagate?"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            s = 1
            t = 1 - s
            return a[t, 0]
        self.check(kernel, a)

    def test_basic32(self):
        """typed int index"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[np.int8(1), 0]
        self.check(kernel, a, expected_exception=[ValueError, LoweringError])

    def test_basic33(self):
        """add 0d array"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[0, 0] + np.array(1)
        self.check(kernel, a)

    def test_basic34(self):
        """More complex rel index with dependency on addition rel index"""
        def kernel(a):
            g = 4. + a[0, 1]
            return g + (a[0, 1] + a[1, 0] + a[0, -1] + np.sin(a[-2, 0]))
        a = np.arange(144).reshape(12, 12)
        self.check(kernel, a)

    # parfors ought to raise but does not, is this likely down to the allocation of the
    # return array under parfors being a different type?
    def test_basic35(self):
        """simple cval """
        def kernel(a):
            return a[0, 1]
        a = np.arange(12.).reshape(3, 4)
        ex = self.exception_dict(stencil=ValueError, parfor=LoweringError, njit=LoweringError)
        self.check(kernel, a, options={'cval': 5}, expected_exception=ex)

    def test_basic36(self):
        """more complex with cval"""
        def kernel(a):
            return a[0, 1] + a[0, -1] + a[1, -1] + a[1, -1]
        a = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, options={'cval': 5.})

    def test_basic37(self):
        """cval is expr"""
        def kernel(a):
            return a[0, 1] + a[0, -1] + a[1, -1] + a[1, -1]
        a = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, options={'cval': 5 + 63.})

    # pyStencil handles the type change, the others raise
    def test_basic38(self):
        """cval is complex"""
        def kernel(a):
            return a[0, 1] + a[0, -1] + a[1, -1] + a[1, -1]
        a = np.arange(12.).reshape(3, 4)
        ex = self.exception_dict(stencil=ValueError, parfor=TypeError, njit=LoweringError)
        self.check(kernel, a, options={'cval': 1.j}, expected_exception=ex)

    def test_basic39(self):
        """cval is func expr"""
        def kernel(a):
            return a[0, 1] + a[0, -1] + a[1, -1] + a[1, -1]
        a = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, options={'cval': np.sin(3.) + np.cos(2)})

    def test_basic40(self):
        """2 args!"""
        def kernel(a, b):
            return a[0, 1] + b[0, -2]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, b)

    # This fails pyStencil but stencil func runs, fairly sure it shouldn't?
    def test_basic41(self):
        """2 args! rel arrays wildly not same size!"""
        def kernel(a, b):
            return a[0, 1] + b[0, -2]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(1.).reshape(1, 1)
        self.check(kernel, a, b, expected_exception=ValueError)

    # This fails pyStencil but stencil func runs, fairly sure it shouldn't?
    def test_basic42(self):
        """2 args! rel arrays very close in size"""
        def kernel(a, b):
            return a[0, 1] + b[0, -2]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(9.).reshape(3, 3)
        self.check(kernel, a, b, expected_exception=[ValueError, LoweringError])

    def test_basic43(self):
        """2 args more complexity"""
        def kernel(a, b):
            return a[0, 1] + a[1, 2] + b[-2, 0] + b[0, -1]
        a = np.arange(30.).reshape(5, 6)
        b = np.arange(30.).reshape(5, 6)
        self.check(kernel, a, b)

    # doesn't match pyStencil, probably expected
    def test_basic44(self):
        """2 args, has assignment before use"""
        def kernel(a, b):
            a[0, 1] = 12
            return a[0, 1]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, b, expected_exception=[ValueError, LoweringError])

    # doesn't match pyStencil, probably expected
    def test_basic45(self):
        """2 args, has assignment and then cross dependency"""
        def kernel(a, b):
            a[0, 1] = 12
            return a[0, 1] + a[1, 0]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, b, expected_exception=[ValueError, LoweringError])

    # doesn't match pyStencil, probably expected
    def test_basic46(self):
        """2 args, has cross relidx assignment"""
        def kernel(a, b):
            a[0, 1] = b[1, 2]
            return a[0, 1] + a[1, 0]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, b, expected_exception=[ValueError, LoweringError])

    def test_basic47(self):
        """3 args"""
        def kernel(a, b, c):
            return a[0, 1] + b[1, 0] + c[-1, 0]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        c = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, b, c)

    # matches pyStencil, but all ought to fail
    # probably hard to detect?
    def test_basic48(self):
        """2 args, has assignment before use via memory alias"""
        def kernel(a):
            c = a.T
            c[:, :] = 10
            return a[0, 1]
        a = np.arange(12.).reshape(3, 4)
        self.check(kernel, a)


    def test_basic49(self):
        """2 args, standard_indexing on second"""
        def kernel(a, b):
            return a[0, 1] + b[0, 3]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, b, options={'standard_indexing': 'b'})

    # this should raise as b[0, 15] is out of bounds for 'b' under standard_indexing
    def test_basic50(self):
        """2 args, standard_indexing OOB"""
        def kernel(a, b):
            return a[0, 1] + b[0, 15]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, b, options={'standard_indexing': 'b'}, expected_exception=IndexError)

    # not sure this makes sense, what size array should be returned when both args are labelled as
    # operating under standard_indexing? It probably ought to raise
    def test_basic51(self):
        """2 args, standard_indexing, no relidx"""
        def kernel(a, b):
            return a[0, 1] + b[0, 2]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, b, options={'standard_indexing': ['a', 'b']}, expected_exception=ValueError)

    def test_basic52(self):
        """3 args, standard_indexing on middle arg """
        def kernel(a, b, c):
            return a[0, 1] + b[0, 1] + c[1, 2]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(4.).reshape(2, 2)
        c = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, b, c, options={'standard_indexing': 'b'})

    # this should fail parfors with a more informative error message
    # @stencil and @njit do not raise but they should
    def test_basic53(self):
        """2 args, standard_indexing on variable that does not exist"""
        def kernel(a, b):
            return a[0, 1] + b[0, 2]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        ex = self.exception_dict(pyStencil=ValueError, stencil=Exception, parfor=KeyError, njit=Exception)
        self.check(kernel, a, b, options={'standard_indexing': 'c'}, expected_exception=ex)

    def test_basic54(self):
        """2 args, standard_indexing, index from var"""
        def kernel(a, b):
            t = 2
            return a[0, 1] + b[0, t]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, b, options={'standard_indexing': 'b'})
        
    def test_basic55(self):
        """2 args, standard_indexing, index from more complex var"""
        def kernel(a, b):
            s = 1
            t = 2 - s
            return a[0, 1] + b[0, t]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, b, options={'standard_indexing': 'b'})
        
    def test_basic56(self):
        """2 args, standard_indexing, added complexity """
        def kernel(a, b):
            s = 1
            acc = 0 
            for k in b[0, :]:
                acc += k
            t = 2 - s - 1
            return a[0, 1] + b[0, t] + acc
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, b, options={'standard_indexing': 'b'})
        
    def test_basic57(self):
        """2 args, standard_indexing, split index operation """
        def kernel(a, b):
            c = b[0]
            return a[0, 1] + c[1]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, b, options={'standard_indexing': 'b'})

    # parfors fails
    def test_basic58(self):
        """2 args, standard_indexing, split index with broadcast mutation """
        def kernel(a, b):
            c = b[0] + 1
            return a[0, 1] + c[1]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, b, options={'standard_indexing': 'b'})

    def test_basic59(self):
        """3 args, mix of array, relative and standard indexing and const"""
        def kernel(a, b, c):
            return a[0, 1] + b[1, 1] + c
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        c = 10
        self.check(kernel, a, b, c, options={'standard_indexing': ['b', 'c']})

    # parfors fails
    def test_basic60(self):
        """3 args, mix of array, relative and standard indexing, tuple pass through"""
        def kernel(a, b, c):
            return a[0, 1] + b[1, 1] + c[0]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        c = (10,)
        self.check(kernel, a, b, c, options={'standard_indexing': ['b', 'c']})

    def test_basic61(self):
        """2 args, standard_indexing on first"""
        def kernel(a, b):
            return a[0, 1] + b[1, 1]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, b, options={'standard_indexing': 'a'})

    def test_basic62(self):
        """2 args, standard_indexing and cval"""
        def kernel(a, b):
            return a[0, 1] + b[1, 1]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, b, options={'standard_indexing': 'b', 'cval': 10.})

    def test_basic63(self):
        """2 args, standard_indexing applied to relative, should fail, non-const idx"""
        def kernel(a, b):
            return a[0, b[0, 1]]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12).reshape(3, 4)
        ex = self.exception_dict(pyStencil=ValueError, stencil=ValueError, parfor=ValueError, njit=LoweringError)
        self.check(kernel, a, b, options={'standard_indexing': 'b'}, expected_exception=ex)
        
    # stencil, njit, parfors all fail. Does this make sense?
    def test_basic64(self):
        """1 arg that uses standard_indexing"""
        def kernel(a):
            return a[0, 0]
        a = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, options={'standard_indexing': 'a'}, expected_exception=ValueError)


    # TODO: neighbourhood

if __name__ == "__main__":
    unittest.main()
