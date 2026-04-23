#![feature(rustc_private)]
extern crate rustc_data_structures;
extern crate rustc_hir;
extern crate rustc_infer;
extern crate rustc_middle;
extern crate rustc_trait_selection;

use flux_rustc_bridge::lowering::resolve_call_query;
use rustc_data_structures::fx::FxIndexMap;
use rustc_hash::{FxHashMap, FxHashSet};
use rustc_hir::{def::DefKind, def_id::DefId};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::{
    mir::TerminatorKind,
    ty::{TyCtxt, TypingMode},
};
use rustc_trait_selection::traits::SelectionContext;

#[derive(Debug, Clone)]
pub struct CallGraph {
    pub inner: FxIndexMap<DefId, Vec<DefId>>,
}

impl CallGraph {
    pub fn new() -> Self {
        Self { inner: FxIndexMap::default() }
    }

    pub fn insert(&mut self, k: DefId, v: Vec<DefId>) -> Option<Vec<DefId>> {
        self.inner.insert(k, v)
    }

    pub fn contains_key(&self, k: &DefId) -> bool {
        self.inner.contains_key(k)
    }

    pub fn get(&self, k: &DefId) -> Option<&Vec<DefId>> {
        self.inner.get(k)
    }

    pub fn merge(&mut self, other: CallGraph) {
        other.inner.into_iter().for_each(|(k, v)| {
            let entries = self.inner.entry(k).or_default();
            for callee in v {
                if !entries.contains(&callee) {
                    entries.push(callee);
                }
            }
        });
    }

    /// Gets all transitive callees from a set of roots
    pub fn transitive_callees(&self, roots: &[DefId]) -> Vec<DefId> {
        let mut result = Vec::new();
        let mut seen = FxHashSet::default();
        let mut worklist: Vec<DefId> = roots.to_vec();

        while let Some(def_id) = worklist.pop() {
            if !seen.insert(def_id) {
                continue;
            }
            result.push(def_id);
            if let Some(callees) = self.get(&def_id) {
                for &callee in callees {
                    if !seen.contains(&callee) {
                        worklist.push(callee);
                    }
                }
            }
        }

        result
    }

    /// Gets all of the paths upwards from the start through callers to a top level fn
    pub fn get_all_paths_from_to(&self, start: DefId, end: DefId) -> Vec<Vec<DefId>> {
        let mut results = Vec::new();
        // initialize with start node
        let mut stack: Vec<Vec<DefId>> = vec![vec![start]];

        while let Some(path) = stack.pop() {
            let current = *path.last().unwrap();

            if current == end {
                // Reached the sink, record the path
                results.push(path);
                continue;
            }

            let callees = self
                .inner
                .get(&current)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);

            if callees.is_empty() {
                // Reached a root without hitting end, discard this path
                continue;
            }

            for &callee in callees {
                if path.contains(&callee) {
                    flux_common::bug!(
                        "flux-cont::call-graph: detected a cycle for {:?}. Recursion is not handled by continuation checking",
                        callee
                    );
                }
                let mut new_path = path.clone();
                new_path.push(callee);
                stack.push(new_path);
            }
        }

        results
    }

    /// Gets all of the paths upwards from the start through callers to a top level fn
    pub fn callers_paths(&self, start: DefId) -> Vec<Vec<DefId>> {
        // build a reverse map from fns to a list of their callers
        let mut reverse: FxHashMap<DefId, Vec<DefId>> = FxHashMap::default();
        for (caller, callees) in &self.inner {
            for callee in callees {
                reverse.entry(*callee).or_default().push(*caller);
            }
        }

        // perform DFS on the reverse map to get all paths from the start fn to the roots
        let mut results = Vec::new();
        let initial_callers = reverse.get(&start).map(|v| v.as_slice()).unwrap_or(&[]);
        let mut stack: Vec<Vec<DefId>> = initial_callers.iter().map(|&c| vec![c]).collect();

        while let Some(path) = stack.pop() {
            let current = *path.last().unwrap();
            let callers = reverse.get(&current).map(|v| v.as_slice()).unwrap_or(&[]);

            if callers.is_empty() {
                // Reached a root (nothing calls this), record the path
                results.push(path);
            } else {
                for &caller in callers {
                    // check that there's no cycle
                    if path.contains(&caller) {
                        flux_common::bug!(
                            "flux-cont::call-graph: detected a cycle for {:?}. Recursion is not handled by continuation checking",
                            caller
                        );
                    }
                    let mut new_path = path.clone();
                    new_path.push(caller);
                    stack.push(new_path);
                }
            }
        }

        results
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CannotResolveReason {
    NoMIRAvailable(DefId, DefKind),
    UnresolvedTraitMethod(DefId),
    NotFnDef { caller: DefId, callee_ty: String },
}

#[derive(Debug, Clone)]
pub struct GraphBuildResult {
    pub call_graph: CallGraph,
    pub resolution_failures: FxHashMap<DefId, CannotResolveReason>,
}

/// Builds the call graph starting from the root function. If we encounter a call we can't resolve, we add it to the resolution_failures map and keep going.
pub fn build_call_graph(tcx: TyCtxt, roots: &[DefId]) -> GraphBuildResult {
    let mut resolution_failures = FxHashMap::default();
    let mut call_graph: CallGraph = CallGraph::new();

    if roots.iter().any(|root| !tcx.def_kind(*root).is_fn_like()) {
        flux_common::bug!(
            "flux-cont::call-graph: all root DefIds must be functions, but found non-function roots: {:?}",
            roots
                .iter()
                .filter(|root| !tcx.def_kind(**root).is_fn_like())
                .map(|root| tcx.def_path_str(*root))
                .collect::<Vec<_>>()
        );
    }

    explore(tcx, roots, &mut call_graph, &mut resolution_failures);

    GraphBuildResult { call_graph, resolution_failures }
}

/// Try to resolve a trait call to a concrete impl. Returns the resolved impl
/// DefId when the impl overrides the method; otherwise returns the original
/// (trait) DefId so the default method body is still reachable in the call graph.
fn resolve_or_fallback<'tcx>(
    tcx: &TyCtxt<'tcx>,
    caller_def_id: DefId,
    def_id: DefId,
    args: rustc_middle::ty::GenericArgsRef<'tcx>,
) -> DefId {
    let param_env = tcx.param_env(caller_def_id);
    let infcx = tcx
        .infer_ctxt()
        .with_next_trait_solver(true)
        .build(TypingMode::non_body_analysis());
    let mut selcx = SelectionContext::new(&infcx);

    match resolve_call_query(*tcx, &mut selcx, param_env, def_id, args) {
        Some((impl_id, _)) => impl_id,
        // Default method, generic parameter, or otherwise unresolvable.
        // Fall back to the trait method DefId — its MIR (if any) contains
        // the default body, which is what we need for sinks like `.send()`.
        None => def_id,
    }
}

/// Returns the callees of a function. Unresolved trait calls fall back to the
/// trait method DefId rather than being dropped, so default-method bodies
/// remain reachable in the call graph.
fn get_callees(tcx: &TyCtxt, caller_def_id: DefId) -> (Vec<DefId>, Vec<CannotResolveReason>) {
    if !tcx.is_mir_available(caller_def_id) || !tcx.def_kind(caller_def_id).is_fn_like() {
        return (vec![], vec![]);
    }

    let body = tcx.optimized_mir(caller_def_id);
    let mut callees = Vec::new();
    let mut failures = Vec::new();

    for bb in body.basic_blocks.iter() {
        // Pick up coroutine bodies produced by async blocks / generators.
        for stmt in &bb.statements {
            if let rustc_middle::mir::StatementKind::Assign(assign) = &stmt.kind {
                let (_place, rvalue) = &**assign;
                if let rustc_middle::mir::Rvalue::Aggregate(kind, _) = rvalue {
                    if let rustc_middle::mir::AggregateKind::Coroutine(def_id, ..) = &**kind {
                        callees.push(*def_id);
                    }
                }
            }
        }

        if let TerminatorKind::Call { func, .. } = &bb.terminator().kind {
            let ty = func.ty(&body.local_decls, *tcx);

            match ty.kind() {
                rustc_middle::ty::TyKind::FnDef(def_id, args) => {
                    let resolved = if tcx.trait_of_assoc(*def_id).is_some() {
                        resolve_or_fallback(tcx, caller_def_id, *def_id, args)
                    } else {
                        *def_id
                    };
                    callees.push(resolved);
                }
                rustc_middle::ty::TyKind::Closure(def_id, _)
                | rustc_middle::ty::TyKind::Coroutine(def_id, _) => {
                    callees.push(*def_id);
                }
                _ => {
                    let ty_str = format!("{:?}", ty);
                    failures.push(CannotResolveReason::NotFnDef {
                        caller: caller_def_id,
                        callee_ty: ty_str,
                    });
                }
            }
        }
    }

    (callees, failures)
}

/// Explores the call graph starting from the root function, populating the call graph and resolution failures.
fn explore(
    tcx: TyCtxt,
    roots: &[DefId],
    call_graph: &mut CallGraph,
    resolution_failures: &mut FxHashMap<DefId, CannotResolveReason>,
) {
    let mut worklist: Vec<DefId> = Vec::new();

    // 1. Seed with roots
    for root in roots {
        let root = *root;
        if !tcx.is_mir_available(root) {
            let def_kind = tcx.def_kind(root);
            if matches!(def_kind, DefKind::AssocFn) {
                resolution_failures.insert(root, CannotResolveReason::UnresolvedTraitMethod(root));
            } else {
                resolution_failures
                    .insert(root, CannotResolveReason::NoMIRAvailable(root, def_kind));
            }
            call_graph.insert(root, Vec::new());
            continue;
        }

        let (callees, failures) = get_callees(&tcx, root);
        call_graph.insert(root, callees);
        for failure in failures {
            resolution_failures.insert(root, failure);
        }
        worklist.push(root);
    }

    // 2. Explore reachable callees
    while let Some(f) = worklist.pop() {
        let callees = call_graph.get(&f).unwrap().clone();

        for callee in callees {
            if call_graph.contains_key(&callee) {
                continue;
            }

            if !tcx.is_mir_available(callee) {
                resolution_failures.insert(
                    callee,
                    CannotResolveReason::NoMIRAvailable(callee, tcx.def_kind(callee)),
                );
                continue;
            }

            let (callee_callees, failures) = get_callees(&tcx, callee);
            call_graph.insert(callee, callee_callees);
            for failure in failures {
                resolution_failures.insert(callee, failure);
            }
            worklist.push(callee);
        }
    }
}
