from __future__ import print_function, division, absolute_import
from collections import namedtuple

import types as pytypes # avoid confusion with numba.types

import numba
from numba import ir, analysis, types, typing, config, numpy_support, cgutils
from numba.ir_utils import (mk_unique_var, replace_vars_inner, find_topo_order,
                            dprint_func_ir, remove_dead, mk_alloc, get_global_func_typ)

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
        self._size_var = None
        self._dist_analysis = None
        self._g_dist_var = None

    def run(self):
        dprint_func_ir(self.func_ir, "starting distributed pass")
        self._dist_analysis = self._analyze_dist()
        if config.DEBUG_ARRAY_OPT==1:
            print("distributions: ", self._dist_analysis)
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
        # add initializations
        first_block = self.func_ir.blocks[topo_order[0]]
        # set scope and loc of generated code to the first variable in block
        scope = first_block.body[0].target.scope
        loc = first_block.body[0].target.loc
        dist_inits = self._get_dist_inits(scope, loc)
        first_block.body = dist_inits+first_block.body

        #
        for label in topo_order:
            new_body = []
            for inst in self.func_ir.blocks[label].body:
                if isinstance(inst, Parfor):
                    new_body += self._run_parfor(inst)
                    continue
                new_body.append(inst)
            self.func_ir.blocks[label].body = new_body

    def _get_dist_inits(self, scope, loc):
        out = []
        # g_dist_var = Global(numba.distributed)
        g_dist_var = ir.Var(scope, mk_unique_var("$distributed_g_var"), loc)
        self._g_dist_var = g_dist_var
        self.typemap[g_dist_var.name] = types.misc.Module(numba.distributed)
        g_dist = ir.Global('distributed', numba.distributed, loc)
        g_dist_assign = ir.Assign(g_dist, g_dist_var, loc)
        # attr call: rank_attr = getattr(g_dist_var, get_rank)
        rank_attr_call = ir.Expr.getattr(g_dist_var, "get_rank", loc)
        rank_attr_var = ir.Var(scope, mk_unique_var("$get_rank_attr"), loc)
        self.typemap[rank_attr_var.name] = get_global_func_typ(get_rank)
        rank_attr_assign = ir.Assign(rank_attr_call, rank_attr_var, loc)
        # rank_var = numba.distributed.get_rank()
        rank_var = ir.Var(scope, mk_unique_var("$rank"), loc)
        self.typemap[rank_var.name] = types.int32
        rank_call = ir.Expr.call(rank_attr_var, [], (), loc)
        self.calltypes[rank_call] = self.typemap[rank_attr_var.name].get_call_type(
            typing.Context(), [], {})
        rank_assign = ir.Assign(rank_call, rank_var, loc)
        self._rank_var = rank_var
        out += [g_dist_assign, rank_attr_assign, rank_assign]

        # attr call: size_attr = getattr(g_dist_var, get_size)
        size_attr_call = ir.Expr.getattr(g_dist_var, "get_size", loc)
        size_attr_var = ir.Var(scope, mk_unique_var("$get_size_attr"), loc)
        self.typemap[size_attr_var.name] = get_global_func_typ(get_size)
        size_attr_assign = ir.Assign(size_attr_call, size_attr_var, loc)
        # size_var = numba.distributed.get_size()
        size_var = ir.Var(scope, mk_unique_var("$dist_size"), loc)
        self.typemap[size_var.name] = types.int32
        size_call = ir.Expr.call(size_attr_var, [], (), loc)
        self.calltypes[size_call] = self.typemap[size_attr_var.name].get_call_type(
            typing.Context(), [], {})
        size_assign = ir.Assign(size_call, size_var, loc)
        self._size_var = size_var
        out += [size_attr_assign, size_assign]
        return out

    def _run_assign(self):
        return

    def _run_parfor(self, parfor):
        if self._dist_analysis.parfor_dists[parfor.id]!=Distribution.OneD:
            if config.DEBUG_ARRAY_OPT==1:
                print("parfor "+str(parfor.id)+" not parallelized.")
            return [parfor]
        #
        scope = parfor.init_block.scope
        loc = parfor.init_block.loc
        out = []
        range_size = parfor.loop_nests[0].range_variable

        div_var = ir.Var(scope, mk_unique_var("$loop_div_var"), loc)
        self.typemap[div_var.name] = types.int64
        div_expr = ir.Expr.binop('//', range_size, self._size_var, loc)
        div_assign = ir.Assign(div_expr, div_var, loc)

        start_var = ir.Var(scope, mk_unique_var("$loop_start_var"), loc)
        self.typemap[start_var.name] = types.int64
        start_expr = ir.Expr.binop('*', div_var, self._rank_var, loc)
        start_assign = ir.Assign(start_expr, start_var, loc)
        # TODO: start loop iteration
        # attr call: end_attr = getattr(g_dist_var, get_end)
        end_attr_call = ir.Expr.getattr(self._g_dist_var, "get_end", loc)
        end_attr_var = ir.Var(scope, mk_unique_var("$get_end_attr"), loc)
        self.typemap[end_attr_var.name] = get_global_func_typ(get_end)
        end_attr_assign = ir.Assign(end_attr_call, end_attr_var, loc)

        end_var = ir.Var(scope, mk_unique_var("$loop_end_var"), loc)
        self.typemap[end_var.name] = types.int64
        end_expr = ir.Expr.call(end_attr_var, [range_size, div_var,
            self._size_var, self._rank_var], (), loc)
        self.calltypes[end_expr] = self.typemap[end_attr_var.name].get_call_type(
            typing.Context(), [types.int64, types.int64, types.int32, types.int32], {})
        end_assign = ir.Assign(end_expr, end_var, loc)
        out += [div_assign, start_assign, end_attr_assign, end_assign]
        out.append(parfor)
        return out

    def _isarray(self, varname):
        return isinstance(self.typemap[varname], types.npytypes.Array)

from numba.typing.templates import infer_global, AbstractTemplate
from numba.typing import signature

def get_rank():
    """dummy function for C mpi get_rank"""
    return 0

def get_size():
    """dummy function for C mpi get_size"""
    return 0

def get_end(total_size, div, pes, rank):
    """get end point of range for parfor division"""
    return total_size if rank==pes-1 else (rank+1)*div

@infer_global(get_rank)
class DistRank(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==0
        return signature(types.int32, *args)

@infer_global(get_size)
class DistSize(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==0
        return signature(types.int32, *args)

@infer_global(get_end)
class DistEnd(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==4
        return signature(types.int64, *args)

from llvmlite import ir as lir

@lower_builtin(get_rank)
def dist_get_rank(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(32), [])
    fn = builder.module.get_or_insert_function(fnty, name="numba_dist_get_rank")
    return builder.call(fn, [])

@lower_builtin(get_size)
def dist_get_size(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(32), [])
    fn = builder.module.get_or_insert_function(fnty, name="numba_dist_get_size")
    return builder.call(fn, [])

@lower_builtin(get_end)
def dist_get_end(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(64), [])
    # TODO: add vars
    fn = builder.module.get_or_insert_function(fnty, name="numba_dist_get_end")
    return builder.call(fn, [])
