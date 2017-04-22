from __future__ import print_function, division, absolute_import
import types as pytypes # avoid confusion with numba.types

from numba import ir, analysis, types, config
from numba.ir_utils import (mk_unique_var, replace_vars_inner, find_topo_order,
                            dprint_func_ir, remove_dead)

from numba.targets.imputils import lower_builtin

import h5py

class PIO(object):
    """analyze and transform hdf5 calls"""
    def __init__(self, func_ir):
        self.func_ir = func_ir
        self.h5_globals = []
        self.h5_file_calls = []
        self.h5_files = []
        self.h5_dsets = {}

    def run(self):
        dprint_func_ir(self.func_ir, "starting IO")
        topo_order = find_topo_order(self.func_ir.blocks)
        for label in topo_order:
            new_body = []
            for inst in self.func_ir.blocks[label].body:
                if isinstance(inst, ir.Assign):
                    inst_list = self._run_assign(inst)
                    if inst_list is not None:
                        new_body.extend(inst_list)
                else:
                    new_body.append(inst)
            self.func_ir.blocks[label].body = new_body
        remove_dead(self.func_ir.blocks, self.func_ir.arg_names)
        dprint_func_ir(self.func_ir, "after IO")
        if config.DEBUG_ARRAY_OPT==1:
            print("h5 files: ", self.h5_files)
            print("h5 dsets: ", self.h5_dsets)

    def _run_assign(self, assign):
        lhs = assign.target.name
        rhs = assign.value
        if (isinstance(rhs, ir.Global) and isinstance(rhs.value, pytypes.ModuleType)
                    and rhs.value==h5py):
            self.h5_globals.append(lhs)
            #return None
        if isinstance(rhs, ir.Expr):
            if rhs.op=='getattr' and rhs.value.name in self.h5_globals and rhs.attr=='File':
                self.h5_file_calls.append(lhs)
                # TODO: return file open call
                #return ir.Expr()
                #return None
            if rhs.op=='call' and rhs.func.name in self.h5_file_calls:
                self.h5_files.append(lhs)
            if rhs.op=='static_getitem' and rhs.value.name in self.h5_files:
                self.h5_dsets[lhs] = (rhs.value, rhs.index_var)
            if rhs.op=='static_getitem' and rhs.value.name in self.h5_dsets:
                f_id, dset  = self.h5_dsets[rhs.value.name]
                loc = rhs.value.loc
                scope = rhs.value.scope
                # TODO: generate size, alloc calls
                # g_pio_var = Global(numba.pio)
                g_pio_var = ir.Var(scope, mk_unique_var("$np_g_var"), loc)
                g_pio = ir.Global('pio', numba.pio, loc)
                g_pio_assign = ir.Assign(g_pio, g_pio_var, loc)
                # attr call: h5size_attr = getattr(g_pio_var, h5size)
                h5size_attr_call = ir.Expr.getattr(g_pio_var, "h5size", loc)
                attr_var = ir.Var(scope, mk_unique_var("$h5size_attr"), loc)
                attr_assign = ir.Assign(h5size_attr_call, attr_var, loc)

                #assign.value = ir.Expr.getattr(numba.pio, 'h5_read', rhs.loc)
                assign.value = ir.Expr.call(attr_var, [f_id, dset], (), rhs.loc)
                return [g_pio_assign, attr_assign, assign]
        # handle copies lhs = rhs
        if isinstance(rhs, ir.Var) and rhs.name in self.h5_files:
            self.h5_files.append(lhs)
        return [assign]

from numba.typing.templates import infer_global, AbstractTemplate
from numba.typing import signature

def h5size():
    """dummy function for C h5_size"""
    pass

@infer_global(h5py.File)
class H5File(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==2
        return signature(types.int32, *args)

@infer_global(h5size)
class H5Size(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==2
        return signature(types.int64, *args)

from llvmlite import ir as lir

#@lower_builtin(h5py.File, types.string, types.string)
#@lower_builtin(h5py.File, types.string, types.Const)
#@lower_builtin(h5py.File, types.Const, types.string)
@lower_builtin(h5py.File, types.Const, types.Const)
def h5_open(context, builder, sig, args):
    # works for constant strings only
    # TODO: extend to string variables
    arg1, arg2 = sig.args
    val1 = context.insert_const_string(builder.module, arg1.value)
    val2 = context.insert_const_string(builder.module, arg2.value)
    fnty = lir.FunctionType(lir.IntType(32), [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="numba_h5_open")
    return builder.call(fn, [val1, val2])

@lower_builtin(h5size, types.int32, types.Const)
def h5_size(context, builder, sig, args):
    # works for constant string only
    # TODO: extend to string variables
    arg1, arg2 = sig.args
    val2 = context.insert_const_string(builder.module, arg2.value)
    fnty = lir.FunctionType(lir.IntType(64), [lir.IntType(32), lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="numba_h5_size")
    return builder.call(fn, [args[0], val2])
