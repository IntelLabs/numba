from __future__ import print_function, division, absolute_import
import types as pytypes # avoid confusion with numba.types

from numba import ir, analysis, types, config
from numba.ir_utils import (mk_unique_var, replace_vars_inner, find_topo_order,
                            dprint_func_ir)

from numba.targets.imputils import lower_builtin

import h5py
import pdb

class PIO(object):
    """analyze and transform hdf5 calls"""
    def __init__(self, func_ir):
        self.func_ir = func_ir
        self.h5_globals = []
        self.h5_file_calls = []
        self.h5_files = []
        self.h5_dsets = []

    def run(self):
        dprint_func_ir(self.func_ir, "starting IO")
        topo_order = find_topo_order(self.func_ir.blocks)
        for label in topo_order:
            new_body = []
            for inst in self.func_ir.blocks[label].body:
                if isinstance(inst, ir.Assign):
                    inst = self._run_assign(inst)
                if inst is not None:
                    new_body.append(inst)
            self.func_ir.blocks[label].body = new_body
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
                #return None
            if rhs.op=='call' and rhs.func.name in self.h5_file_calls:
                self.h5_files.append(lhs)
            if rhs.op=='static_getitem' and rhs.value.name in self.h5_files:
                self.h5_dsets.append(lhs)
            if rhs.op=='static_getitem' and rhs.value.name in self.h5_dsets:
                print("data: ", lhs)
                # TODO: read call
        # handle copies lhs = rhs
        if isinstance(rhs, ir.Var) and rhs.name in self.h5_files:
            self.h5_files.append(lhs)
        return assign

@lower_builtin('h5_open', types.string, types.string)
def h5_open(context, builder, sig, args):
    #pdb.set_trace()
    return 3
