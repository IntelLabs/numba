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

from enum import Enum
class Distribution(Enum):
    REP = 1
    OneD = 3
    TwoD = 2

_dist_analysis_result = namedtuple('dist_analysis_result', 'array_dists,parfor_dists')

class DistributedPass(object):
    """analyze program and transfrom to distributed"""
    def __init__(self, func_ir, typemap):
        self.func_ir = func_ir
        self.typemap = typemap

    def run(self):
        dprint_func_ir(self.func_ir, "starting distributed pass")
        dist_analysis = self._analyze_dist()
        if config.DEBUG_ARRAY_OPT==1:
            print("distributions: ", dist_analysis)

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

    def _run_assign(self):
        return

    def _run_parfor(self, parfor):
        return

    def _isarray(self, varname):
        return isinstance(self.typemap[varname], types.npytypes.Array)
