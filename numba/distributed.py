from __future__ import print_function, division, absolute_import
from collections import namedtuple

import types as pytypes # avoid confusion with numba.types

import numba
from numba import ir, analysis, types, config, numpy_support, cgutils
from numba.ir_utils import (mk_unique_var, replace_vars_inner, find_topo_order,
                            dprint_func_ir, remove_dead, mk_alloc)

from numba.targets.imputils import lower_builtin
from numba.targets.arrayobj import make_array
from numba.parfor import Parfor
import numpy as np

import h5py
from mpi4py import MPI

from enum import Enum
class Distribution(Enum):
    REP = 1
    OneD = 3
    TwoD = 2

_dist_analysis_result = namedtuple('dist_analysis_result', 'array_dists,parfor_dists')

class DistributedPass(object):
    """analyze program and transfrom to distributed"""
    def __init__(self, func_ir, typemap, calltypes):
        self.func_ir = func_ir
        self.typemap = typemap
        self.calltypes = calltypes
        self._rank_var = None # will be set in run

    def run(self):
        dprint_func_ir(self.func_ir, "starting distributed pass")
        dist_analysis = self._analyze_dist()
        if config.DEBUG_ARRAY_OPT==1:
            print("distributions: ", dist_analysis)
        self._run_dist_pass()
        dprint_func_ir(self.func_ir, "after distributed pass")

    def _analyze_dist(self):
        topo_order = find_topo_order(self.func_ir.blocks)
        array_dists = {}
        parfor_dists = {}
        save_array_dists = {}
        save_parfor_dists = {1:1} # dummy value
        # fixed-point iteration
        while array_dists!=save_array_dists or parfor_dists!=save_parfor_dists:
            for label in topo_order:
                for inst in self.func_ir.blocks[label].body:
                    if isinstance(inst, ir.Assign):
                        self._analyze_assign(inst, array_dists, parfor_dists)
                    elif isinstance(inst, Parfor):
                        self._analyze_parfor(inst, array_dists, parfor_dists)
            save_array_dists = array_dists
            save_parfor_dists = parfor_dists

        return _dist_analysis_result(array_dists=array_dists, parfor_dists=parfor_dists)

    def _analyze_assign(self, inst, array_dists, parfor_dists):
        lhs = inst.target.name
        rhs = inst.value
        if self._isarray(lhs) and lhs not in array_dists:
            array_dists[lhs] = Distribution.OneD

        if isinstance(rhs, ir.Var):
            new_dist = min(array_dists[lhs].value, array_dists[rhs.name].value)
            array_dists[lhs] = Distribution(new_dist)
            array_dists[rhs.name] = Distribution(new_dist)
        return

    def _analyze_parfor(self, parfor, array_dists, parfor_dists):
        if parfor.id not in parfor_dists:
            parfor_dists[parfor.id] = Distribution.OneD
        return

    def _run_dist_pass(self):
        topo_order = find_topo_order(self.func_ir.blocks)
        first_block = self.func_ir.blocks[topo_order[0]]
        # set scope and loc of generated code to the first variable in block
        scope = first_block.body[0].target.scope
        loc = first_block.body[0].target.loc
        dist_inits = self._get_dist_inits(scope, loc)
        first_block.body = dist_inits+first_block.body

    def _get_dist_inits(self, scope, loc):
        out = []
        # g_mpi_var = Global(numba.distributed)
        g_mpi_var = ir.Var(scope, mk_unique_var("$distributed_g_var"), loc)
        self.typemap[g_mpi_var.name] = types.misc.Module(numba.distributed)
        g_mpi = ir.Global('distributed', numba.distributed, loc)
        g_mpi_assign = ir.Assign(g_mpi, g_mpi_var, loc)
        # attr call: rank_attr = getattr(g_mpi_var, get_rank)
        rank_attr_call = ir.Expr.getattr(g_mpi_var, "get_rank", loc)
        rank_attr_var = ir.Var(scope, mk_unique_var("$get_rank_attr"), loc)
        # typemap =
        rank_attr_assign = ir.Assign(rank_attr_call, rank_attr_var, loc)
        # rank_var = numba.distributed.get_rank()
        rank_var = ir.Var(scope, mk_unique_var("$rank"), loc)
        rank_call = ir.Expr.call(rank_attr_var, [], (), loc)
        rank_assign = ir.Assign(rank_call, rank_var, loc)
        self._rank_var = rank_var
        out += [g_mpi_assign, rank_attr_assign, rank_assign]

        return out

    def _run_assign(self):
        return

    def _run_parfor(self, parfor):
        return

    def _isarray(self, varname):
        return isinstance(self.typemap[varname], types.npytypes.Array)
