use std::{collections::HashMap, path::Path};

use flux_common::{bug, cache::QueryCache, iter::IterExt, result::ResultExt};
use flux_config::{self as config};
use flux_errors::FluxSession;
use flux_infer::{
    fixpoint_encoding::{
        FixQueryCache, FixpointCtxt, KVarDecl, KVarEncoding,
        fixpoint::{self, Task},
    },
    infer::Tag,
    lean_encoding,
};
use flux_metadata::CStore;
use flux_middle::{
    Specs,
    def_id::MaybeExternId,
    fhir::{self},
    global_env::GlobalEnv,
    metrics::{self, Metric, TimingKind},
    queries::{Providers, QueryResult},
    rty::{StaticInfo, fold::TypeFoldable},
};
use flux_refineck as refineck;
use liquid_fixpoint::FixpointStatus;
use rustc_borrowck::consumers::ConsumerOptions;
use rustc_data_structures::{
    fx::{FxIndexMap, FxIndexSet},
    snapshot_map::SnapshotMap,
};
use rustc_driver::{Callbacks, Compilation};
use rustc_errors::ErrorGuaranteed;
use rustc_hir::{
    def::{CtorKind, DefKind},
    def_id::{DefId, LOCAL_CRATE, LocalDefId},
};
use rustc_interface::interface::Compiler;
use rustc_middle::{query, ty::TyCtxt};
use rustc_session::config::OutputType;
use rustc_span::FileName;

use crate::{DEFAULT_LOCALE_RESOURCES, collector::SpecCollector};

#[derive(Default)]
pub struct FluxCallbacks;

impl Callbacks for FluxCallbacks {
    fn config(&mut self, config: &mut rustc_interface::interface::Config) {
        assert!(config.override_queries.is_none());

        config.override_queries = Some(|_, local| {
            local.mir_borrowck = mir_borrowck;
        });
        // this should always be empty otherwise something changed in rustc and all our assumptions
        // about symbol interning are wrong.
        assert!(config.extra_symbols.is_empty());
        config.extra_symbols = flux_syntax::symbols::PREDEFINED_FLUX_SYMBOLS.to_vec();
    }

    fn after_analysis(&mut self, compiler: &Compiler, tcx: TyCtxt<'_>) -> Compilation {
        self.verify(compiler, tcx);
        if config::full_compilation() { Compilation::Continue } else { Compilation::Stop }
    }
}

impl FluxCallbacks {
    fn verify(&self, compiler: &Compiler, tcx: TyCtxt<'_>) {
        if compiler.sess.dcx().has_errors().is_some() {
            return;
        }

        let sess = FluxSession::new(
            &tcx.sess.opts,
            tcx.sess.psess.clone_source_map(),
            rustc_errors::fallback_fluent_bundle(DEFAULT_LOCALE_RESOURCES.to_vec(), false),
        );

        let mut providers = Providers::default();
        flux_desugar::provide(&mut providers);
        flux_fhir_analysis::provide(&mut providers);
        providers.collect_specs = collect_specs;

        let cstore = CStore::load(tcx, &sess);
        let arena = fhir::Arena::new();
        GlobalEnv::enter(tcx, &sess, Box::new(cstore), &arena, providers, |genv| {
            let result = metrics::time_it(TimingKind::Total, || check_crate(genv));
            if result.is_ok() {
                encode_and_save_metadata(genv);
            }
            lean_encoding::finalize(genv).unwrap_or(());
        });
        let _ = metrics::print_and_dump_timings(tcx);
        sess.finish_diagnostics();
    }
}

fn check_crate(genv: GlobalEnv) -> Result<(), ErrorGuaranteed> {
    tracing::info_span!("check_crate").in_scope(move || {
        tracing::info!("Callbacks::check_wf");
        // Query qualifiers and spec funcs to report wf errors
        let _ = genv.qualifiers().emit(&genv)?;
        let _ = genv.normalized_defns(LOCAL_CRATE);

        let mut ck = CrateChecker::new(genv);

        let (sources, mut sinks): (FxIndexSet<DefId>, FxIndexSet<DefId>) =
            genv.tcx().iter_local_def_id().fold(
                (FxIndexSet::default(), FxIndexSet::default()),
                |(mut sources, mut sinks), local_def_id| {
                    if genv.tcx().def_kind(local_def_id).is_fn_like() {
                        if genv.is_source(local_def_id) {
                            sources.insert(local_def_id.to_def_id());
                        }
                        if genv.is_sink(local_def_id) {
                            sinks.insert(local_def_id.to_def_id());
                        }
                    }
                    (sources, sinks)
                },
            );

        let mut source_call_graph = flux_cont::CallGraph::new();
        for source in sources.iter() {
            let cg = genv.call_graph(source).expect("Could not build call graph");
            source_call_graph.merge(cg);
        }

        // check non-locals in call graph for sinks
        for (_def_id, callees) in &source_call_graph.inner {
            for callee in callees {
                if genv.is_sink(callee) {
                    sinks.insert(*callee);
                }
            }
        }

        for local_def_id in genv.tcx().iter_local_def_id() {
            let def_id = genv.maybe_extern_id(local_def_id);
            let _ = trigger_queries(genv, def_id);
        }

        let mut crash_log: Vec<(DefId, String)> = Vec::new();
        let mut solution_log: FxIndexMap<String, Vec<(fhir::SinkType, Vec<(_, _)>)>> = FxIndexMap::default();

        for source in sources.iter() {
            // clear def ids and fixpoint context maps for each source
            ck.def_id_to_cstr_map.clear();
            ck.def_id_to_fixpoint_ctx.clear();

            let mut paths_to_check = Vec::new();

            // accumulate set of sinks that should be checked for ** THIS ** specific source
            let mut sinks_to_check = FxIndexSet::default();
            for sink in sinks.iter() {
                let paths = source_call_graph.get_all_paths_from_to(*source, *sink);
                if paths.len() > 0 {
                    sinks_to_check.insert(*sink);
                    paths_to_check.extend(paths);
                }
            }

            let def_ids_to_check =
                paths_to_check
                    .iter()
                    .flatten()
                    .fold(FxIndexSet::default(), |mut acc, def_id| {
                        match def_id.as_local() {
                            Some(local) => {
                                acc.insert(local);
                            }
                            None => {}
                        }
                        acc
                    });

            // get all transitive callees of the functions in the paths
            let all_def_ids: Vec<DefId> = def_ids_to_check.iter().map(|l| l.to_def_id()).collect();
            let additional_callees = source_call_graph.transitive_callees(&all_def_ids);

            let all_to_check: FxIndexSet<LocalDefId> = def_ids_to_check
                .into_iter()
                .chain(
                    additional_callees
                        .into_iter()
                        .filter_map(|def_id| def_id.as_local()),
                )
                .collect();

            let _ = all_to_check
                .into_iter()
                .try_for_each_exhaust(|def_id| ck.check_def_catching_bugs(def_id));

            let mut tasks = ck.def_id_to_cstr_map.clone().into_iter().map(|(_, t)| t);

            if let Some(mut mega_task) = tasks.next() {
                // TODO: Fix merging constants so that they are globally 
                // defined and we don't have potential clashes. 
                // Right now they are all created locally from idx 0
                // and we could have potential mismatches
                for task in tasks {
                    mega_task.merge(task);
                }

                let mut local_sink_kvar_map = HashMap::new();
                for sink in sinks_to_check.iter() {
                    let sink_kvar = genv
                        .get_sink_kvar_for(*sink)
                        .unwrap_or_else(|| bug!("Each sink should have a stored kvar, but sink {sink:?} did not have one."));

                    local_sink_kvar_map.insert(sink, sink_kvar.clone());

                    let fixpoint_kvid = fixpoint::KVid::from_u32(sink_kvar.kvid.as_u32());
                    let local = fixpoint::LocalVar::from_u32(9999);

                    let sort = mega_task
                        .kvars
                        .iter()
                        .find(|k| k.kvid == fixpoint_kvid)
                        .map(|k| k.sorts[0].clone())
                        .unwrap_or_else(|| bug!("Could not get kvar sort for kvar: {fixpoint_kvid:?}"));

                    let bind = fixpoint::Bind {
                        name: fixpoint::Var::Local(local),
                        sort,
                        pred: fixpoint::Pred::KVar(
                            fixpoint_kvid,
                            vec![fixpoint::Expr::Var(fixpoint::Var::Local(local))],
                        ),
                    };

                    let trivial = fixpoint::Constraint::Pred(
                        fixpoint::Pred::Expr(fixpoint::Expr::Atom(
                            fixpoint::BinRel::Lt,
                            Box::new([fixpoint::Expr::int(10), fixpoint::Expr::int(11)]),
                        )),
                        None,
                    );

                    let consumer = fixpoint::Constraint::ForAll(bind, Box::new(trivial));

                    let existing =
                        std::mem::replace(&mut mega_task.constraint, fixpoint::Constraint::TRUE);
                    mega_task.constraint = fixpoint::Constraint::Conj(vec![existing, consumer]);
                }

                // println!("{mega_task}");
                let verification_result = match mega_task.run() {
                    Ok(r) => r,
                    Err(err) => {
                        // println!("{mega_task}");
                        // bug!();
                        crash_log.push((*source, format!("mega_task run failed in mono phase: {err}")));
                        continue;
                    }
                };


                if let FixpointStatus::Crash(ref crash_reason) = verification_result.status {
                        // println!("FAILED FOR {source:?}");
                        // println!("{mega_task}");
                        // bug!();
                    crash_log.push((*source, format!("FixpointStatus::Crash in mono phase: {crash_reason:?}")));
                    continue;
                }

                let map_len = ck.def_id_to_cstr_map.len();
                let mut ctxs = ck.def_id_to_fixpoint_ctx.drain(0..map_len).map(|(_, c)| c);

                if let Some(mut base_fcx) = ctxs.next() {
                    for ctx in ctxs {
                        base_fcx.merge(ctx);
                    }

                    let kvar_solutions =
                        base_fcx.parse_kvar_solutions(&verification_result.non_cuts_solution);

                    for (sink_def_id, sink_kvar) in local_sink_kvar_map.iter() {
                        let kvid = flux_infer::fixpoint_encoding::fixpoint::KVid::from_u32(
                            sink_kvar.kvid.as_u32(),
                        );
                        let sol = kvar_solutions.get(&kvid).unwrap_or_else(|| {
                            bug!("Sink KVar had no solution in mono constraint")
                        });


                        let res = base_fcx.fixpoint_to_solution(sol);
                        let kvar_sort = res.vars()[0].expect_sort().clone();
                        let simplified_sol = res
                            .skip_binder_ref()
                            .simplify(&SnapshotMap::default())
                            .normalize(genv)
                            .eliminate_dead_variables();

                        let disjuncts = simplified_sol.to_dnf();

                        let mut constraints = Vec::new();

                        let disjuncts: Vec<_> = disjuncts
                            .into_iter()
                            .map(|d| d.simplify(&SnapshotMap::default()))
                            .filter(|d| !d.is_trivially_false())
                            .collect();

                        let mut fresh_kvids = Vec::new();
                        for _ in disjuncts.iter() {
                            fresh_kvids.push(genv.get_next_kvid());
                        }

                        for (disjunct, fresh_kvid) in disjuncts.iter().zip(fresh_kvids.iter()) {
                            base_fcx.ecx.local_var_env.push_layer_with_fresh_names(1);

                            let guard = base_fcx
                                .ecx
                                .expr_to_fixpoint(&disjunct, &mut base_fcx.scx)
                                .expect("Could not encode disjunct");

                            let vars = base_fcx.ecx.local_var_env.pop_layer();

                            let sort = base_fcx.scx.sort_to_fixpoint(&kvar_sort);

                            let bind = fixpoint::Bind {
                                name: fixpoint::Var::Local(vars[0]),
                                sort,
                                pred: fixpoint::Pred::Expr(guard),
                            };
                            let head = fixpoint::Constraint::Pred(
                                fixpoint::Pred::KVar(
                                    flux_infer::fixpoint_encoding::fixpoint::KVid::from_u32(
                                        fresh_kvid.as_u32(),
                                    ),
                                    vec![fixpoint::Expr::Var(fixpoint::Var::Local(vars[0]))],
                                ),
                                None,
                            );
                            constraints.push(fixpoint::Constraint::ForAll(bind, Box::new(head)));
                        }

                        for fresh_kvid in &fresh_kvids {
                            base_fcx.kvars.add(
                                *fresh_kvid,
                                KVarDecl {
                                    self_args: 1,
                                    sorts: vec![kvar_sort.clone()],
                                    encoding: KVarEncoding::Single,
                                },
                            );

                            let fixpoint_kvid = fixpoint::KVid::from_u32(fresh_kvid.as_u32());
                            base_fcx
                                .kcx
                                .ranges
                                .insert(*fresh_kvid, fixpoint_kvid..fixpoint_kvid + 1);
                        }

                        for (_, fresh_kvid) in fresh_kvids.iter().enumerate() {
                            let local = base_fcx.ecx.local_var_env.fresh_name();
                            let sort = base_fcx.scx.sort_to_fixpoint(&kvar_sort);
                            let fixpoint_kvid = fixpoint::KVid::from_u32(fresh_kvid.as_u32());

                            let trivial_check = fixpoint::Constraint::Pred(
                                fixpoint::Pred::Expr(fixpoint::Expr::Atom(
                                    fixpoint::BinRel::Lt,
                                    Box::new([fixpoint::Expr::int(10), fixpoint::Expr::int(11)]),
                                )),
                                None,
                            );

                            let kvar_bind = fixpoint::Bind {
                                name: fixpoint::Var::Underscore,
                                sort: sort.clone(),
                                pred: fixpoint::Pred::KVar(
                                    fixpoint_kvid,
                                    vec![fixpoint::Expr::Var(fixpoint::Var::Local(local))],
                                ),
                            };

                            let outer_bind = fixpoint::Bind {
                                name: fixpoint::Var::Local(local),
                                sort,
                                pred: fixpoint::Pred::TRUE,
                            };

                            let consumer = fixpoint::Constraint::ForAll(
                                outer_bind,
                                Box::new(fixpoint::Constraint::ForAll(
                                    kvar_bind,
                                    Box::new(trivial_check),
                                )),
                            );

                            constraints.push(consumer);
                        }

                        let combined = fixpoint::Constraint::Conj(constraints);

                        use rustc_hir::def_id::CRATE_DEF_ID;

                        let dummy_def_id = MaybeExternId::Local(CRATE_DEF_ID);
                        let solver = match genv.infer_opts(source.as_local().unwrap()).solver {
                            flux_config::SmtSolver::Z3 => liquid_fixpoint::SmtSolver::Z3,
                            flux_config::SmtSolver::CVC5 => liquid_fixpoint::SmtSolver::CVC5,
                        };
                        let mut task = base_fcx
                            .create_task(
                                dummy_def_id,
                                combined,
                                false,
                                solver
                            )
                            .expect("Failed to create task");

                        for fresh_kvid in &fresh_kvids {
                            let fixpoint_kvid = fixpoint::KVid::from_u32(fresh_kvid.as_u32());
                            task.add_cut_kvar(fixpoint_kvid);
                        }

                        // println!("{task}");
 
                        // add qualifiers to the task
                        for sink in sinks_to_check.iter() {
                            let sink_for = genv.sink_for(*sink) ;
                                match sink_for {
fhir::SinkType::DynamoPut => {
                                        task.string_qualifiers.push(
                                            "
;; qualifiers for DynamoPut - we want to be able to infer table name and items being set

;; qualifiers to determine the variant of

(qualif DynPutIsBool ((a0 Adt2831635821) (a1# Str))
    (= mkadt1194516581$2 (fld1924543948$0 (Map_select (fld2831635821$1 a0) a1))))

(qualif DynPutIsStr ((a0 Adt2831635821) (a1# Str))
    (= mkadt1194516581$0  (fld1924543948$0 (Map_select (fld2831635821$1 a0) a1))))

(qualif DynPutIsNum ((a0 Adt2831635821) (a1# Str))
    (= mkadt1194516581$1 (fld1924543948$0 (Map_select (fld2831635821$1 a0) a1))))

(qualif DynPutIsBinary ((a0 Adt2831635821) (a1# Str))
    (= mkadt1194516581$3 (fld1924543948$0 (Map_select (fld2831635821$1 a0) a1))))

(qualif DynPutIsBinarySet ((a0 Adt2831635821) (a1# Str))
    (= mkadt1194516581$4 (fld1924543948$0 (Map_select (fld2831635821$1 a0) a1))))

(qualif DynPutIsList ((a0 Adt2831635821) (a1# Str))
    (= mkadt1194516581$5 (fld1924543948$0 (Map_select (fld2831635821$1 a0) a1))))

(qualif DynPutIsMap ((a0 Adt2831635821) (a1# Str))
    (= mkadt1194516581$6 (fld1924543948$0 (Map_select (fld2831635821$1 a0) a1))))

(qualif DynPutIsNumberSet ((a0 Adt2831635821) (a1# Str))
    (= mkadt1194516581$7 (fld1924543948$0 (Map_select (fld2831635821$1 a0) a1))))

(qualif DynPutIsNull ((a0 Adt2831635821) (a1# Str))
    (= mkadt1194516581$8 (fld1924543948$0 (Map_select (fld2831635821$1 a0) a1))))

(qualif DynPutIsStringSet ((a0 Adt2831635821) (a1# Str))
    (= mkadt1194516581$9 (fld1924543948$0 (Map_select (fld2831635821$1 a0) a1))))

(qualif DynPutStrVal ((a0 Adt2831635821) (a1# Str) (a2# Str))
    (= (fld1924543948$1 (Map_select (fld2831635821$1 a0) a1)) a2))

(qualif DynPutBoolValT ((a0 Adt2831635821) (a1# Str))
    (= true (fld1924543948$2 (Map_select (fld2831635821$1 a0) a1))))

(qualif DynPutBoolValF ((a0 Adt2831635821) (a1# Str))
    (= false (fld1924543948$2 (Map_select (fld2831635821$1 a0) a1))))

;; Table name
(qualif DynPutTableName ((a0 Adt2831635821) (a1# Str))
    (= (fld2831635821$0 a0) a1))
                                            "
                                        )
                                    }
                                    fhir::SinkType::DynamoGet => {
                                        task.string_qualifiers.push(
                                            "
;; qualifiers for DynamoGet - we want to be able to infer table name and key

;; qualifiers to determine the variant of

(qualif DynGetIsBool ((a0 Adt309293745) (a1# Str))
    (= mkadt1194516581$2 (fld1924543948$0 (Map_select (fld309293745$1 a0) a1))))

(qualif DynGetIsStr ((a0 Adt309293745) (a1# Str))
    (= mkadt1194516581$0  (fld1924543948$0 (Map_select (fld309293745$1 a0) a1))))

(qualif DynGetIsNum ((a0 Adt309293745) (a1# Str))
    (= mkadt1194516581$1 (fld1924543948$0 (Map_select (fld309293745$1 a0) a1))))

(qualif DynGetIsBinary ((a0 Adt309293745) (a1# Str))
    (= mkadt1194516581$3 (fld1924543948$0 (Map_select (fld309293745$1 a0) a1))))

(qualif DynGetIsBinarySet ((a0 Adt309293745) (a1# Str))
    (= mkadt1194516581$4 (fld1924543948$0 (Map_select (fld309293745$1 a0) a1))))

(qualif DynGetIsList ((a0 Adt309293745) (a1# Str))
    (= mkadt1194516581$5 (fld1924543948$0 (Map_select (fld309293745$1 a0) a1))))

(qualif DynGetIsMap ((a0 Adt309293745) (a1# Str))
    (= mkadt1194516581$6 (fld1924543948$0 (Map_select (fld309293745$1 a0) a1))))

(qualif DynGetIsNumberSet ((a0 Adt309293745) (a1# Str))
    (= mkadt1194516581$7 (fld1924543948$0 (Map_select (fld309293745$1 a0) a1))))

(qualif DynGetIsNull ((a0 Adt309293745) (a1# Str))
    (= mkadt1194516581$8 (fld1924543948$0 (Map_select (fld309293745$1 a0) a1))))

(qualif DynGetIsStringSet ((a0 Adt309293745) (a1# Str))
    (= mkadt1194516581$9 (fld1924543948$0 (Map_select (fld309293745$1 a0) a1))))

(qualif DynGetStrVal ((a0 Adt309293745) (a1# Str) (a2# Str))
    (= (fld1924543948$1 (Map_select (fld309293745$1 a0) a1)) a2))

(qualif DynGetBoolValT ((a0 Adt309293745) (a1# Str))
    (= true (fld1924543948$2 (Map_select (fld309293745$1 a0) a1))))

(qualif DynGetBoolValF ((a0 Adt309293745) (a1# Str))
    (= false (fld1924543948$2 (Map_select (fld309293745$1 a0) a1))))

;; Table name
(qualif DynGetTableName ((a0 Adt309293745) (a1# Str))
    (= (fld309293745$0 a0) a1))
                                            "
                                        )
                                    }
                                    fhir::SinkType::DynamoDelete => {
                                        task.string_qualifiers.push(
                                            "
;; qualifiers for DynamoDelete - we want to be able to infer table name and key

;; qualifiers to determine the variant of

(qualif DynDeleteIsBool ((a0 Adt1346103878) (a1# Str))
    (= mkadt1194516581$2 (fld1924543948$0 (Map_select (fld1346103878$1 a0) a1))))

(qualif DynDeleteIsStr ((a0 Adt1346103878) (a1# Str))
    (= mkadt1194516581$0  (fld1924543948$0 (Map_select (fld1346103878$1 a0) a1))))

(qualif DynDeleteIsNum ((a0 Adt1346103878) (a1# Str))
    (= mkadt1194516581$1 (fld1924543948$0 (Map_select (fld1346103878$1 a0) a1))))

(qualif DynDeleteIsBinary ((a0 Adt1346103878) (a1# Str))
    (= mkadt1194516581$3 (fld1924543948$0 (Map_select (fld1346103878$1 a0) a1))))

(qualif DynDeleteIsBinarySet ((a0 Adt1346103878) (a1# Str))
    (= mkadt1194516581$4 (fld1924543948$0 (Map_select (fld1346103878$1 a0) a1))))

(qualif DynDeleteIsList ((a0 Adt1346103878) (a1# Str))
    (= mkadt1194516581$5 (fld1924543948$0 (Map_select (fld1346103878$1 a0) a1))))

(qualif DynDeleteIsMap ((a0 Adt1346103878) (a1# Str))
    (= mkadt1194516581$6 (fld1924543948$0 (Map_select (fld1346103878$1 a0) a1))))

(qualif DynDeleteIsNumberSet ((a0 Adt1346103878) (a1# Str))
    (= mkadt1194516581$7 (fld1924543948$0 (Map_select (fld1346103878$1 a0) a1))))

(qualif DynDeleteIsNull ((a0 Adt1346103878) (a1# Str))
    (= mkadt1194516581$8 (fld1924543948$0 (Map_select (fld1346103878$1 a0) a1))))

(qualif DynDeleteIsStringSet ((a0 Adt1346103878) (a1# Str))
    (= mkadt1194516581$9 (fld1924543948$0 (Map_select (fld1346103878$1 a0) a1))))

(qualif DynDeleteStrVal ((a0 Adt1346103878) (a1# Str) (a2# Str))
    (= (fld1924543948$1 (Map_select (fld1346103878$1 a0) a1)) a2))

(qualif DynDeleteBoolValT ((a0 Adt1346103878) (a1# Str))
    (= true (fld1924543948$2 (Map_select (fld1346103878$1 a0) a1))))

(qualif DynDeleteBoolValF ((a0 Adt1346103878) (a1# Str))
    (= false (fld1924543948$2 (Map_select (fld1346103878$1 a0) a1))))

;; Table name
(qualif DynDeleteTableName ((a0 Adt1346103878) (a1# Str))
    (= (fld1346103878$0 a0) a1))
                                            "
                                        )
                                    }
                                    fhir::SinkType::DynamoUpdate => {
                                        task.string_qualifiers.push(
                                            "
;; qualifiers for DynamoUpdate - we want to be able to infer table name and key

;; qualifiers to determine the variant of

(qualif DynUpdateIsBool ((a0 Adt139055347) (a1# Str))
    (= mkadt1194516581$2 (fld1924543948$0 (Map_select (fld139055347$1 a0) a1))))

(qualif DynUpdateIsStr ((a0 Adt139055347) (a1# Str))
    (= mkadt1194516581$0  (fld1924543948$0 (Map_select (fld139055347$1 a0) a1))))

(qualif DynUpdateIsNum ((a0 Adt139055347) (a1# Str))
    (= mkadt1194516581$1 (fld1924543948$0 (Map_select (fld139055347$1 a0) a1))))

(qualif DynUpdateIsBinary ((a0 Adt139055347) (a1# Str))
    (= mkadt1194516581$3 (fld1924543948$0 (Map_select (fld139055347$1 a0) a1))))

(qualif DynUpdateIsBinarySet ((a0 Adt139055347) (a1# Str))
    (= mkadt1194516581$4 (fld1924543948$0 (Map_select (fld139055347$1 a0) a1))))

(qualif DynUpdateIsList ((a0 Adt139055347) (a1# Str))
    (= mkadt1194516581$5 (fld1924543948$0 (Map_select (fld139055347$1 a0) a1))))

(qualif DynUpdateIsMap ((a0 Adt139055347) (a1# Str))
    (= mkadt1194516581$6 (fld1924543948$0 (Map_select (fld139055347$1 a0) a1))))

(qualif DynUpdateIsNumberSet ((a0 Adt139055347) (a1# Str))
    (= mkadt1194516581$7 (fld1924543948$0 (Map_select (fld139055347$1 a0) a1))))

(qualif DynUpdateIsNull ((a0 Adt139055347) (a1# Str))
    (= mkadt1194516581$8 (fld1924543948$0 (Map_select (fld139055347$1 a0) a1))))

(qualif DynUpdateIsStringSet ((a0 Adt139055347) (a1# Str))
    (= mkadt1194516581$9 (fld1924543948$0 (Map_select (fld139055347$1 a0) a1))))

(qualif DynUpdateStrVal ((a0 Adt139055347) (a1# Str) (a2# Str))
    (= (fld1924543948$1 (Map_select (fld139055347$1 a0) a1)) a2))

(qualif DynUpdateBoolValT ((a0 Adt139055347) (a1# Str))
    (= true (fld1924543948$2 (Map_select (fld139055347$1 a0) a1))))

(qualif DynUpdateBoolValF ((a0 Adt139055347) (a1# Str))
    (= false (fld1924543948$2 (Map_select (fld139055347$1 a0) a1))))

;; Table name
(qualif DynUpdateTableName ((a0 Adt139055347) (a1# Str))
    (= (fld139055347$0 a0) a1))

;; Update expression
(qualif DynUpdateUpdateExpression ((a0 Adt139055347) (a1# Str))
    (= (fld139055347$2 a0) a1))

;; Expression attribute names
(qualif DynUpdateExprAttrName ((a0 Adt139055347) (a1# Str) (a2# Str))
    (= (Map_select (fld139055347$3 a0) a1) a2))

;; Expression attribute values - type tags
(qualif DynUpdateExprAttrIsStr ((a0 Adt139055347) (a1# Str))
    (= mkadt1194516581$0 (fld1924543948$0 (Map_select (fld139055347$4 a0) a1))))

(qualif DynUpdateExprAttrIsNum ((a0 Adt139055347) (a1# Str))
    (= mkadt1194516581$1 (fld1924543948$0 (Map_select (fld139055347$4 a0) a1))))

(qualif DynUpdateExprAttrIsBool ((a0 Adt139055347) (a1# Str))
    (= mkadt1194516581$2 (fld1924543948$0 (Map_select (fld139055347$4 a0) a1))))

(qualif DynUpdateExprAttrIsBinary ((a0 Adt139055347) (a1# Str))
    (= mkadt1194516581$3 (fld1924543948$0 (Map_select (fld139055347$4 a0) a1))))

(qualif DynUpdateExprAttrIsBinarySet ((a0 Adt139055347) (a1# Str))
    (= mkadt1194516581$4 (fld1924543948$0 (Map_select (fld139055347$4 a0) a1))))

(qualif DynUpdateExprAttrIsList ((a0 Adt139055347) (a1# Str))
    (= mkadt1194516581$5 (fld1924543948$0 (Map_select (fld139055347$4 a0) a1))))

(qualif DynUpdateExprAttrIsMap ((a0 Adt139055347) (a1# Str))
    (= mkadt1194516581$6 (fld1924543948$0 (Map_select (fld139055347$4 a0) a1))))

(qualif DynUpdateExprAttrIsNumberSet ((a0 Adt139055347) (a1# Str))
    (= mkadt1194516581$7 (fld1924543948$0 (Map_select (fld139055347$4 a0) a1))))

(qualif DynUpdateExprAttrIsNull ((a0 Adt139055347) (a1# Str))
    (= mkadt1194516581$8 (fld1924543948$0 (Map_select (fld139055347$4 a0) a1))))

(qualif DynUpdateExprAttrIsStringSet ((a0 Adt139055347) (a1# Str))
    (= mkadt1194516581$9 (fld1924543948$0 (Map_select (fld139055347$4 a0) a1))))

;; Expression attribute values - concrete values
(qualif DynUpdateExprAttrStrVal ((a0 Adt139055347) (a1# Str) (a2# Str))
    (= (fld1924543948$1 (Map_select (fld139055347$4 a0) a1)) a2))

(qualif DynUpdateExprAttrBoolValT ((a0 Adt139055347) (a1# Str))
    (= true (fld1924543948$2 (Map_select (fld139055347$4 a0) a1))))

(qualif DynUpdateExprAttrBoolValF ((a0 Adt139055347) (a1# Str))
    (= false (fld1924543948$2 (Map_select (fld139055347$4 a0) a1))))
                                            "
                                        )
                                    }
                                    fhir::SinkType::DynamoQuery => {
                                        task.string_qualifiers.push(
                                            "
;; qualifiers for DynamoQuery - we want to be able to infer table name and key

;; Consistent read
(qualif DynQueryConsistentReadT ((a0 Adt4204386769))
    (= true (fld4204386769$1 a0)))

(qualif DynQueryConsistentReadF ((a0 Adt4204386769))
    (= false (fld4204386769$1 a0)))

;; Index name
(qualif DynQueryIndexName ((a0 Adt4204386769) (a1# Str))
    (= (fld4204386769$2 a0) a1))

;; Condition expression
(qualif DynQueryConditionExpression ((a0 Adt4204386769) (a1# Str))
    (= (fld4204386769$3 a0) a1))

;; Filter expression
(qualif DynQueryFilterExpression ((a0 Adt4204386769) (a1# Str))
    (= (fld4204386769$4 a0) a1))

;; Expression attribute names
(qualif DynQueryExprAttrName ((a0 Adt4204386769) (a1# Str) (a2# Str))
    (= (Map_select (fld4204386769$5 a0) a1) a2))

;; Expression attribute values - type tags
(qualif DynQueryExprAttrIsStr ((a0 Adt4204386769) (a1# Str))
    (= mkadt1194516581$0 (fld1924543948$0 (Map_select (fld4204386769$6 a0) a1))))

(qualif DynQueryExprAttrIsNum ((a0 Adt4204386769) (a1# Str))
    (= mkadt1194516581$1 (fld1924543948$0 (Map_select (fld4204386769$6 a0) a1))))

(qualif DynQueryExprAttrIsBool ((a0 Adt4204386769) (a1# Str))
    (= mkadt1194516581$2 (fld1924543948$0 (Map_select (fld4204386769$6 a0) a1))))

(qualif DynQueryExprAttrIsBinary ((a0 Adt4204386769) (a1# Str))
    (= mkadt1194516581$3 (fld1924543948$0 (Map_select (fld4204386769$6 a0) a1))))

(qualif DynQueryExprAttrIsBinarySet ((a0 Adt4204386769) (a1# Str))
    (= mkadt1194516581$4 (fld1924543948$0 (Map_select (fld4204386769$6 a0) a1))))

(qualif DynQueryExprAttrIsList ((a0 Adt4204386769) (a1# Str))
    (= mkadt1194516581$5 (fld1924543948$0 (Map_select (fld4204386769$6 a0) a1))))

(qualif DynQueryExprAttrIsMap ((a0 Adt4204386769) (a1# Str))
    (= mkadt1194516581$6 (fld1924543948$0 (Map_select (fld4204386769$6 a0) a1))))

(qualif DynQueryExprAttrIsNumberSet ((a0 Adt4204386769) (a1# Str))
    (= mkadt1194516581$7 (fld1924543948$0 (Map_select (fld4204386769$6 a0) a1))))

(qualif DynQueryExprAttrIsNull ((a0 Adt4204386769) (a1# Str))
    (= mkadt1194516581$8 (fld1924543948$0 (Map_select (fld4204386769$6 a0) a1))))

(qualif DynQueryExprAttrIsStringSet ((a0 Adt4204386769) (a1# Str))
    (= mkadt1194516581$9 (fld1924543948$0 (Map_select (fld4204386769$6 a0) a1))))

;; Expression attribute values - concrete values
(qualif DynQueryExprAttrStrVal ((a0 Adt4204386769) (a1# Str) (a2# Str))
    (= (fld1924543948$1 (Map_select (fld4204386769$6 a0) a1)) a2))

(qualif DynQueryExprAttrBoolValT ((a0 Adt4204386769) (a1# Str))
    (= true (fld1924543948$2 (Map_select (fld4204386769$6 a0) a1))))

(qualif DynQueryExprAttrBoolValF ((a0 Adt4204386769) (a1# Str))
    (= false (fld1924543948$2 (Map_select (fld4204386769$6 a0) a1))))

;; Table name
(qualif DynQueryTableName ((a0 Adt4204386769) (a1# Str))
    (= (fld4204386769$0 a0) a1))
                                            "
                                        )
                                    }
                                    fhir::SinkType::S3PutObject => {
                                        task.string_qualifiers.push(
                                            "
;; qualifiers for S3Put - we want to be able to infer bucket name and object key

;; Bucket name
(qualif S3PutBucketName ((a0 Adt2834044593) (a1# Str))
    (= (fld2834044593$0 a0) a1))

;; Object Key
(qualif S3PutObjectKey ((a0 Adt2834044593) (a1# Str))
    (= (fld2834044593$1 a0) a1))
                                            "
                                        )
                                    }
                                    fhir::SinkType::S3GetObject => {
                                        task.string_qualifiers.push(
                                            "
(qualif S3GetBucketName ((a0 Adt2571902642) (a1# Str))
    (= (fld2571902642$0 a0) a1))

(qualif S3GetObjectKey ((a0 Adt2571902642) (a1# Str))
    (= (fld2571902642$1 a0) a1))
                                            "
                                        )
                                    }
                                    fhir::SinkType::S3DeleteObject => {
                                        task.string_qualifiers.push(
                                            "
;; qualifiers for S3Delete - we want to be able to infer bucket name and object key

;; Bucket name
(qualif S3DeleteBucketName ((a0 Adt942972120) (a1# Str))
    (= (fld942972120$0 a0) a1))

;; Object Key
(qualif S3DeleteObjectKey ((a0 Adt942972120) (a1# Str))
    (= (fld942972120$1 a0) a1))
                                            "
                                        )
                                    }
                                    fhir::SinkType::Unknown => {}
                                }
                        }

                        let verification_result = match task.run() {
                            Ok(r) => r,
                            Err(err) => {
                        // bug!();
                                crash_log.push((*source, format!("per-sink task run failed in cut phase: {err}")));
                                continue;
                            }
                        };

                        if let FixpointStatus::Crash(ref crash_reason) = verification_result.status {
                        // println!("{mega_task}");
                        // bug!();
                            crash_log.push((*source, format!("FixpointStatus::Crash in cut phase: {crash_reason:?}")));
                            continue;
                        }

                        let cut_kvar_solutions =
                            base_fcx.parse_kvar_solutions(&verification_result.solution);

                        let mut solutions = Vec::new();
                        for (kvar_id, sol) in cut_kvar_solutions.iter() {
                            let res = base_fcx.fixpoint_to_solution(sol);
                            solutions.push((*kvar_id, res));
                        }

                        let sink_for = genv.sink_for(*sink_def_id);
                        let source_assoc_names = genv.source_for(source);
                        for name in source_assoc_names {
                            match solution_log.get_mut(&name) {
                                Some(v) => {
                                    v.push((sink_for, solutions.clone()));
                                }
                                None => {
                                    solution_log.insert(name, vec![(sink_for, solutions.clone())]);
                                }
                            };
                        }
                    }
                }
            }
        }

        // ── Print solutions ───────────────────────────────────
        for (source, entries) in solution_log {
            println!("SOLUTION FOR {:?}", source);
            for (sink_for, solutions) in entries {
                println!("{sink_for:?}");
                for (kvar_id, res) in solutions {
                    println!("{kvar_id:?}: {:#?}", res);
                }
                println!();
            }
        }

        // ── Print crash summary ───────────────────────────────────────────────────────
        println!("=== CRASH SUMMARY ({} crashed) ===", crash_log.len());
        for (source, reason) in &crash_log {
            println!("  source={:?}  reason={}", source, reason);
            println!();
        }
        println!("=== END CRASH SUMMARY ===");

        // if config::lean().is_check() || config::lean().is_emit() {
        //     lean_encoding::finalize(genv)
        //         .unwrap_or_else(|err| bug!("error running lean-check {err:?}"));
        // }

        // let lean_result = if config::lean().is_check() {
        //     genv.iter_local_def_id().try_for_each_exhaust(|def_id| {
        //         if genv.proven_externally(def_id).is_some() {
        //             let key = lean_task_key(genv.tcx(), def_id.to_def_id());
        //             // Skip proof check if previously verified successfully.
        //             if config::is_cache_enabled()
        //                 && ck
        //                     .cache
        //                     .lookup_by_key(&key)
        //                     .map(|r| matches!(r.lean_status, LeanStatus::Valid))
        //                     .unwrap_or(false)
        //             {
        //                 return Ok(());
        //             }
        //             lean_encoding::check_proof(genv, def_id.to_def_id())?;
        //             // Mark as valid in cache so future runs skip re-verification.
        //             ck.cache
        //                 .update_result_by_key(&key, |r| r.lean_status = LeanStatus::Valid);
        //             Ok(())
        //         } else {
        //             Ok(())
        //         }
        //     })
        // } else {
        //     Ok(())
        // };

        // ck.cache.save().unwrap_or(());

        tracing::info!("Callbacks::check_crate");

        Ok(())
    })
}

fn collect_specs(genv: GlobalEnv) -> Specs {
    match SpecCollector::collect(genv.tcx(), genv.sess()) {
        Ok(specs) => specs,
        Err(err) => {
            genv.sess().abort(err);
        }
    }
}

fn encode_and_save_metadata(genv: GlobalEnv) {
    // We only save metadata when `--emit=metadata` is passed as an argument. In this case, we save
    // the `.fluxmeta` file alongside the `.rmeta` file. This setup works for `cargo flux`, which
    // wraps `cargo check` and always passes `--emit=metadata`. Tests also explicitly pass this flag.
    let tcx = genv.tcx();
    if tcx
        .output_filenames(())
        .outputs
        .contains_key(&OutputType::Metadata)
    {
        let path = flux_metadata::filename_for_metadata(tcx);
        flux_metadata::encode_metadata(genv, path.as_path());
    }
}

struct CrateChecker<'genv, 'tcx> {
    genv: GlobalEnv<'genv, 'tcx>,
    cache: FixQueryCache,
    def_id_to_cstr_map: FxIndexMap<DefId, Task>,
    def_id_to_fixpoint_ctx: FxIndexMap<DefId, FixpointCtxt<'genv, 'tcx, Tag>>,
}

impl<'genv, 'tcx> CrateChecker<'genv, 'tcx> {
    fn new(genv: GlobalEnv<'genv, 'tcx>) -> Self {
        Self {
            genv,
            cache: QueryCache::load(),
            def_id_to_cstr_map: FxIndexMap::default(),
            def_id_to_fixpoint_ctx: FxIndexMap::default(),
        }
    }

    fn matches_def(&self, def_id: MaybeExternId, def: &str) -> bool {
        // Does this def_id's name contain `fn_name`?
        let def_path = self.genv.tcx().def_path_str(def_id.local_id());
        def_path.contains(def)
    }

    fn matches_file_path<F>(&self, def_id: MaybeExternId, matcher: F) -> bool
    where
        F: Fn(&Path) -> bool,
    {
        let def_id = def_id.local_id();
        let tcx = self.genv.tcx();
        let span = tcx.def_span(def_id);
        let sm = tcx.sess.source_map();
        let FileName::Real(file_name) = sm.span_to_filename(span) else { return true };
        let mut file_path = file_name.local_path_if_available();

        // If the path is absolute try to normalize it to be relative to the working_dir
        if file_path.is_absolute() {
            let working_dir = tcx.sess.opts.working_dir.local_path_if_available();
            let Ok(p) = file_path.strip_prefix(working_dir) else { return true };
            file_path = p;
        }

        matcher(file_path)
    }

    fn matches_pos(&self, def_id: MaybeExternId, line: usize, col: usize) -> bool {
        let def_id = def_id.local_id();
        let tcx = self.genv.tcx();
        let hir_id = tcx.local_def_id_to_hir_id(def_id);
        let body_span = tcx.hir_span_with_body(hir_id);
        let source_map = tcx.sess.source_map();
        let lo_pos = source_map.lookup_char_pos(body_span.lo());
        let start_line = lo_pos.line;
        let start_col = lo_pos.col_display;
        let hi_pos = source_map.lookup_char_pos(body_span.hi());
        let end_line = hi_pos.line;
        let end_col = hi_pos.col_display;

        // is the line in the range of the body?
        if start_line < end_line {
            // multiple lines: check if the line is in the range
            start_line <= line && line <= end_line
        } else {
            // single line: check if the line is the same and the column is in range
            start_line == line && start_col <= col && col <= end_col
        }
    }

    /// Check whether the `def_id` (or the file where `def_id` is defined)
    /// is in the `include` pattern, and conservatively return `true` if
    /// anything unexpected happens.
    fn is_included(&self, def_id: MaybeExternId) -> bool {
        let Some(pattern) = config::include_pattern() else { return true };
        if self.matches_file_path(def_id, |path| pattern.glob.is_match(path)) {
            return true;
        }
        if pattern.defs.iter().any(|def| self.matches_def(def_id, def)) {
            return true;
        }
        if pattern.spans.iter().any(|pos| {
            self.matches_file_path(def_id, |path| path.ends_with(&pos.file))
                && self.matches_pos(def_id, pos.line, pos.column)
        }) {
            return true;
        }
        false
    }

    fn check_def_catching_bugs(&mut self, def_id: LocalDefId) -> Result<(), ErrorGuaranteed> {
        let mut this = std::panic::AssertUnwindSafe(self);
        let msg = format!("def_id: {:?}, span: {:?}", def_id, this.genv.tcx().def_span(def_id));
        flux_common::bug::catch_bugs(&msg, move || this.check_def(def_id))?
    }

    fn check_def(&mut self, def_id: LocalDefId) -> Result<(), ErrorGuaranteed> {
        let genv = self.genv;
        let def_id = genv.maybe_extern_id(def_id);

        // Dummy items generated for extern specs are excluded from metrics
        if genv.is_dummy(def_id.local_id()) {
            return Ok(());
        }

        let kind = genv.def_kind(def_id);

        // For the purpose of metrics, we consider to be a *function* an item that
        // 1. It's local, i.e., it's not an extern spec.
        // 2. It's a free function (`DefKind::Fn`) or associated item (`DefKind::AssocFn`), and
        // 3. It has a mir body
        // In particular, this excludes closures (because they dont have the right `DefKind`) and
        // trait methods without a default body.
        let is_fn_with_body = def_id
            .as_local()
            .map(|local_id| {
                matches!(kind, DefKind::Fn | DefKind::AssocFn)
                    && genv.tcx().is_mir_available(local_id)
            })
            .unwrap_or(false);

        metrics::incr_metric_if(is_fn_with_body, Metric::FnTotal);

        if genv.ignored(def_id.local_id()) {
            metrics::incr_metric_if(is_fn_with_body, Metric::FnIgnored);
            return Ok(());
        }
        if !self.is_included(def_id) {
            metrics::incr_metric_if(is_fn_with_body, Metric::FnTrusted);
            return Ok(());
        }

        trigger_queries(genv, def_id).emit(&genv)?;

        match kind {
            DefKind::Fn | DefKind::AssocFn => {
                let Some(local_id) = def_id.as_local() else { return Ok(()) };
                if is_fn_with_body {
                    refineck::check_fn(
                        genv,
                        &mut self.cache,
                        local_id,
                        &mut self.def_id_to_cstr_map,
                        &mut self.def_id_to_fixpoint_ctx,
                    )?;
                }
            }
            DefKind::Enum => {
                let adt_def = genv.adt_def(def_id).emit(&genv)?;
                let enum_def = genv
                    .fhir_expect_item(def_id.local_id())
                    .emit(&genv)?
                    .expect_enum();
                refineck::invariants::check_invariants(
                    genv,
                    &mut self.cache,
                    def_id,
                    enum_def.invariants,
                    &adt_def,
                    &mut self.def_id_to_cstr_map,
                    &mut self.def_id_to_fixpoint_ctx,
                )?;
            }
            DefKind::Struct => {
                // We check invariants for `struct` in `check_constructor` (i.e. when the struct is built),
                // so nothing to do here.
            }
            DefKind::Impl { of_trait } => {
                if of_trait {
                    refineck::compare_impl_item::check_impl_against_trait(genv, def_id)
                        .emit(&genv)?;
                }
            }
            DefKind::TyAlias => {}
            DefKind::Trait => {}
            DefKind::Static { .. } => {
                if let StaticInfo::Known(ty) = genv.static_info(def_id).emit(&genv)?
                    && let Some(local_id) = def_id.as_local()
                {
                    refineck::check_static(
                        genv,
                        &mut self.cache,
                        local_id,
                        ty,
                        &mut self.def_id_to_cstr_map,
                        &mut self.def_id_to_fixpoint_ctx,
                    )?;
                }
            }
            _ => (),
        }
        Ok(())
    }
}

/// Triggers queries for the given `def_id` to mark it as "reached" for metadata encoding.
///
/// This function ensures that all relevant queries for a definition are triggered upfront,
/// so the item and its associated data will be included in the encoded metadata. Without this,
/// items might be missing from the metadata (extern specs in particular which are not otherwise "checked"),
/// causing errors when dependent crates try to use them.
fn trigger_queries(genv: GlobalEnv, def_id: MaybeExternId) -> QueryResult {
    match genv.def_kind(def_id) {
        DefKind::Trait => {
            genv.generics_of(def_id)?;
            genv.predicates_of(def_id)?;
            genv.refinement_generics_of(def_id)?;
        }
        DefKind::Impl { .. } => {
            genv.generics_of(def_id)?;
            genv.predicates_of(def_id)?;
            genv.refinement_generics_of(def_id)?;
        }
        DefKind::Fn | DefKind::AssocFn => {
            genv.generics_of(def_id)?;
            genv.refinement_generics_of(def_id)?;
            genv.predicates_of(def_id)?;
            genv.fn_sig(def_id)?;
        }
        DefKind::Ctor(_, CtorKind::Fn) => {
            genv.generics_of(def_id)?;
            genv.refinement_generics_of(def_id)?;
            // We don't report the error because it can raise a `QueryErr::OpaqueStruct`,  which
            // should be reported at the use site.
            let _ = genv.fn_sig(def_id);
        }
        DefKind::Enum | DefKind::Struct => {
            genv.generics_of(def_id)?;
            genv.predicates_of(def_id)?;
            genv.refinement_generics_of(def_id)?;
            genv.adt_def(def_id)?;
            genv.adt_sort_def_of(def_id)?;
            genv.variants_of(def_id)?;
            genv.type_of(def_id)?;
        }
        DefKind::TyAlias => {
            genv.generics_of(def_id)?;
            genv.predicates_of(def_id)?;
            genv.refinement_generics_of(def_id)?;
            genv.type_of(def_id)?;
        }
        DefKind::OpaqueTy => {
            genv.generics_of(def_id)?;
            genv.predicates_of(def_id)?;
            genv.item_bounds(def_id)?;
            genv.refinement_generics_of(def_id)?;
        }
        _ => {}
    }
    Ok(())
}

fn mir_borrowck<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> query::queries::mir_borrowck::ProvidedValue<'tcx> {
    let bodies_with_facts = rustc_borrowck::consumers::get_bodies_with_borrowck_facts(
        tcx,
        def_id,
        ConsumerOptions::RegionInferenceContext,
    );
    for (def_id, body_with_facts) in bodies_with_facts {
        // SAFETY: This is safe because we are feeding in the same `tcx` that is
        // going to be used as a witness when pulling out the data.
        unsafe {
            flux_common::mir_storage::store_mir_body(tcx, def_id, body_with_facts);
        }
    }
    let mut providers = query::Providers::default();
    rustc_borrowck::provide(&mut providers);
    let original_mir_borrowck = providers.mir_borrowck;
    original_mir_borrowck(tcx, def_id)
}
