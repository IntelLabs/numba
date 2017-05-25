from __future__ import print_function

import sys

import numba.unittest_support as unittest
from numba import njit, jit, testing, utils
from .support import TestCase

# import numpy in two ways, both uses needed
import numpy as np
import numpy


class TestClosure(TestCase):

    def run_jit_closure_variable(self, **jitargs):
        Y = 10

        def add_Y(x):
            return x + Y

        c_add_Y = jit('i4(i4)', **jitargs)(add_Y)
        self.assertEqual(c_add_Y(1), 11)

        # Like globals in Numba, the value of the closure is captured
        # at time of JIT
        Y = 12  # should not affect function
        self.assertEqual(c_add_Y(1), 11)

    def test_jit_closure_variable(self):
        self.run_jit_closure_variable(forceobj=True)

    def test_jit_closure_variable_npm(self):
        self.run_jit_closure_variable(nopython=True)

    def run_rejitting_closure(self, **jitargs):
        Y = 10

        def add_Y(x):
            return x + Y

        c_add_Y = jit('i4(i4)', **jitargs)(add_Y)
        self.assertEqual(c_add_Y(1), 11)

        # Redo the jit
        Y = 12
        c_add_Y_2 = jit('i4(i4)', **jitargs)(add_Y)
        self.assertEqual(c_add_Y_2(1), 13)
        Y = 13  # should not affect function
        self.assertEqual(c_add_Y_2(1), 13)

        self.assertEqual(c_add_Y(1), 11)  # Test first function again

    def test_rejitting_closure(self):
        self.run_rejitting_closure(forceobj=True)

    def test_rejitting_closure_npm(self):
        self.run_rejitting_closure(nopython=True)

    def run_jit_multiple_closure_variables(self, **jitargs):
        Y = 10
        Z = 2

        def add_Y_mult_Z(x):
            return (x + Y) * Z

        c_add_Y_mult_Z = jit('i4(i4)', **jitargs)(add_Y_mult_Z)
        self.assertEqual(c_add_Y_mult_Z(1), 22)

    def test_jit_multiple_closure_variables(self):
        self.run_jit_multiple_closure_variables(forceobj=True)

    def test_jit_multiple_closure_variables_npm(self):
        self.run_jit_multiple_closure_variables(nopython=True)

    def run_jit_inner_function(self, **jitargs):
        def mult_10(a):
            return a * 10

        c_mult_10 = jit('intp(intp)', **jitargs)(mult_10)
        c_mult_10.disable_compile()

        def do_math(x):
            return c_mult_10(x + 4)

        c_do_math = jit('intp(intp)', **jitargs)(do_math)
        c_do_math.disable_compile()

        with self.assertRefCount(c_do_math, c_mult_10):
            self.assertEqual(c_do_math(1), 50)

    def test_jit_inner_function(self):
        self.run_jit_inner_function(forceobj=True)

    def test_jit_inner_function_npm(self):
        self.run_jit_inner_function(nopython=True)

    @testing.allow_interpreter_mode
    def test_return_closure(self):

        def outer(x):

            def inner():
                return x + 1

            return inner

        cfunc = jit(outer)
        self.assertEqual(cfunc(10)(), outer(10)())


class TestInlinedClosure(TestCase):
    """
    Tests for (partial) closure support in njit. The support is partial
    because it only works for closures that can be successfully inlined
    at compile time.
    """

    def test_inner_function(self):

        def outer(x):

            def inner(x):
                return x * x

            return inner(x) + inner(x)

        cfunc = njit(outer)
        self.assertEqual(cfunc(10), outer(10))

    def test_inner_function_with_closure(self):

        def outer(x):
            y = x + 1

            def inner(x):
                return x * x + y

            return inner(x) + inner(x)

        cfunc = njit(outer)
        self.assertEqual(cfunc(10), outer(10))

    def test_inner_function_with_closure_2(self):

        def outer(x):
            y = x + 1

            def inner(x):
                return x * y

            y = inner(x)
            return y + inner(x)

        cfunc = njit(outer)
        self.assertEqual(cfunc(10), outer(10))

    @unittest.skipIf(utils.PYVERSION < (3, 0), "needs Python 3")
    def test_inner_function_with_closure_3(self):

        code = """
            def outer(x):
                y = x + 1
                z = 0

                def inner(x):
                    nonlocal z
                    z += x * x
                    return z + y

                return inner(x) + inner(x) + z
        """
        ns = {}
        exec(code.strip(), ns)

        cfunc = njit(ns['outer'])
        self.assertEqual(cfunc(10), ns['outer'](10))

    def test_inner_function_nested(self):

        def outer(x):

            def inner(y):

                def innermost(z):
                    return x + y + z

                s = 0
                for i in range(y):
                    s += innermost(i)
                return s

            return inner(x * x)

        cfunc = njit(outer)
        self.assertEqual(cfunc(10), outer(10))


class TestVariousPR2369(TestCase):

    def test_many(self):

        def outer1(x):
            """ Test calling recursive function from inner """
            def inner(x):
                return fib3(x)
            return inner(x)

        def outer2(x):
            """ Test calling recursive function from closure """
            z = x + 1

            def inner(x):
                return x + fib3(z)
            return inner(x)

        def outer3(x):
            """ Test recursive inner """
            def inner(x):
                if x + y < 2:
                    return 10
                else:
                    inner(x - 1)
            return inner(x)

        def outer4(x):
            """ Test recursive closure """
            y = x + 1

            def inner(x):
                if x + y < 2:
                    return 10
                else:
                    inner(x - 1)
            return inner(x)

        def outer5(x):
            """ Test nested closure """
            y = x + 1

            def inner1(x):
                z = y + x + 2

                def inner2(x):
                    return x + z

                return inner2(x) + y

            return inner1(x)

        def outer6(x):
            """ Test closure with list comprehension in body """
            y = x + 1

            def inner1(x):
                z = y + x + 2
                return [t for t in range(z)]
            return inner1(x)

        _OUTER_SCOPE_VAR = 9

        def outer7(x):
            """ Test use of outer scope var, no closure """
            z = x + 1
            return x + z + _OUTER_SCOPE_VAR

        _OUTER_SCOPE_VAR = 9

        def outer8(x):
            """ Test use of outer scope var, with closure """
            z = x + 1

            def inner(x):
                return x + z + _OUTER_SCOPE_VAR
            return inner(x)

        def outer9(x):
            """ Test closure assignment"""
            z = x + 1

            def inner(x):
                return x + z
            f = inner
            return f(x)

        def outer10(x):
            """ Test two inner, one calls other """
            z = x + 1

            def inner(x):
                return x + z

            def inner2(x):
                return inner(x)

            return inner2(x)

        def outer11(x):
            """ return the closure """
            z = x + 1

            def inner(x):
                return x + z
            return inner

        def outer12(x):
            """ closure with kwarg"""
            z = x + 1

            def inner(x, kw=7):
                return x + z + kw
            return inner

        def outer13(x, kw=7):
            """ outer with kwarg no closure"""
            z = x + 1 + kw
            return z

        def outer14(x, kw=7):
            """ outer with kwarg used in closure"""
            z = x + 1

            def inner(x):
                return x + z + kw
            return inner

        def outer15(x, kw=7):
            """ outer with kwarg as arg to closure"""
            z = x + 1

            def inner(x, kw):
                return x + z + kw
            return inner

        def list1(x):
            """ Test basic list comprehension """
            return [i for i in range(1, len(x) - 1)]

        def list2(x):
            """ Test conditional list comprehension """
            return [y for y in x if y < 2]

        def list3(x):
            """ Test ternary list comprehension """
            return [y if y < 2 else -1 for y in x]

        def list4(x):
            """ Test list comprehension to np.array ctor """
            return np.array([1, 2, 3])

        # expected fail, unsupported type in sequence
        def list5(x):
            """ Test nested list comprehension to np.array ctor """
            return np.array([np.array([z for z in x]) for y in x])

        def list6(x):
            """ Test use of inner function in list comprehension """
            def inner(x):
                return x + 1
            return [inner(z) for z in x]

        def list7(x):
            """ Test use of closure in list comprehension """
            y = 3

            def inner(x):
                return x + y
            return [inner(z) for z in x]

        def list8(x):
            """ Test use of list comprehension as arg to inner function """
            l = [z + 1 for z in x]

            def inner(x):
                return x[0] + 1
            q = inner(l)
            return q

        def list9(x):
            """ Test use of list comprehension access in closure """
            l = [z + 1 for z in x]

            def inner(x):
                return x[0] + l[1]
            return inner(x)

        def list10(x):
            """ Test use of list comprehension access in closure and as arg """
            l = [z + 1 for z in x]

            def inner(x):
                return [y + l[0] for y in x]
            return inner(l)

        # expected fail, nested mem managed object
        def list11(x):
            """ Test scalar array construction in list comprehension """
            l = [np.array(z) for z in x]
            return l

        def list12(x):
            """ Test scalar type conversion construction in list comprehension """
            l = [np.float64(z) for z in x]
            return l

        def list13(x):
            """ Test use of explicit numpy scalar ctor reference in list comprehension """
            l = [numpy.float64(z) for z in x]
            return l

        def list14(x):
            """ Test use of python scalar ctor reference in list comprehension """
            l = [float(z) for z in x]
            return l

        def list15(x):
            """ Test use of python scalar ctor reference in list comprehension followed by np array construction from the list"""
            l = [float(z) for z in x]
            return np.array(l)

        def list16(x):
            """ Test type unification from np array ctors consuming list comprehension """
            l1 = [float(z) for z in x]
            l2 = [z for z in x]
            ze = np.array(l1)
            oe = np.array(l2)
            return ze + oe

        def list17(x):
            """ Test complex list comprehension including math calls """
            return [(a, b, c)
                    for a in x for b in x for c in x if np.sqrt(a**2 + b**2) == c]

        _OUTER_SCOPE_VAR = 9

        def list18(x):
            """ Test loop list with outer scope var as conditional"""
            z = []
            for i in x:
                if i < _OUTER_SCOPE_VAR:
                    z.append(i)
            return z

        _OUTER_SCOPE_VAR = 9

        def list19(x):
            """ Test list comprehension with outer scope as conditional"""
            return [i for i in x if i < _OUTER_SCOPE_VAR]

        def list20(x):
            """ Test return empty list """
            return [i for i in x if i == -1000]

        def list21(x):
            """ Test call a jitted function in a list comprehension """
            return [fib3(i) for i in x]

        def list22(x):
            """ Test create two lists comprehensions and a third walking the first two """
            a = [y - 1 for y in x]
            b = [y + 1 for y in x]
            return [x for x in a for y in b if x == y]

        def list23(x):
            """ Test operation on comprehension generated list """
            z = [y for y in x]
            z.append(1)
            return z

        # basic test/debug loops

        # closure/inner func cases
        f = [outer1, outer2, outer3, outer4, outer5,
             outer6, outer7, outer8, outer9, outer10,
             outer11, outer12, outer13, outer14, outer15]
        for ref in f:
            cfunc = njit(ref)
            var = 10
            try:
                self.assertEqual(cfunc(var), ref(var))
            except Exception as e:
                print("fail: %s" % ref)
                print("%s\n\n" % e)
            else:
                pass
                #print("pass: %s" % ref)

        # list comp
        f = [list1, list2, list3, list4, list5,
             list6, list7, list8, list9, list10,
             list11, list12, list13, list14, list15,
             list16, list17, list18, list19, list20,
             list21, list22, list23]
        for ref in f:
            var = [1, 2, 3, 4, 5]
            try:
                cfunc = njit(ref)
                self.assertEqual(cfunc(var), ref(var))
            except ValueError as e:  # likely np array returned
                try:
                    np.testing.assert_allclose(cfunc(var), ref(var))
                except Exception as e:
                    print("fail: %s" % ref)
                    print("%s\n\n" % e)
                else:
                    pass
                    #print("pass: %s" % ref)
            except Exception as e:
                print("fail: %s" % ref)
                print("%s\n\n" % e)
            else:
                pass
                #print("pass: %s" % ref)


@njit
def fib3(n):
    if n < 2:
        return n
    return fib3(n - 1) + fib3(n - 2)


if __name__ == '__main__':
    unittest.main()
