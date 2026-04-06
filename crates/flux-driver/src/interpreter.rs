// Imports you'll need — adjust paths to match your crate layout
extern crate rustc_type_ir;

use flux_infer::fixpoint_encoding::fixpoint::ThyFunc;
use flux_middle::rty::{
    AliasReft, BinOp, Binder, BoundReftKind, Constant, Ctor, ESpan, Expr, ExprKind, FieldProj,
    HoleKind, InternalFuncKind, KVar, Lambda, Loc, Name, Path, SortArg, SpecFuncKind, UnOp, Var,
};

// ── Value ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Bool(bool),
    Int(i128),
    Str(String),
    Char(char),
    Tuple(Vec<Value>),
    /// (def_id display name, variant_idx, fields)
    Adt(String, usize, Vec<Value>),
    /// Finite map: association list
    Map(Vec<(Value, Value)>),
    /// A lambda we can't reduce yet (returned as-is)
    Abs(Lambda),
    /// Anything we can't evaluate: holes, kvars, alias refts, unresolved vars
    Opaque(String),
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Bool(b) => write!(f, "{b}"),
            Value::Int(n) => write!(f, "{n}"),
            Value::Str(s) => write!(f, "\"{s}\""),
            Value::Char(c) => write!(f, "'{c}'"),
            Value::Abs(_) => write!(f, "<lambda>"),
            Value::Opaque(s) => write!(f, "<{s}>"),
            Value::Tuple(vs) => {
                write!(f, "(")?;
                for (i, v) in vs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, ")")
            }
            Value::Adt(name, variant, fields) => {
                write!(f, "{name}::{variant} {{ ")?;
                for (i, v) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, ".{i} = {v}")?;
                }
                write!(f, " }}")
            }
            Value::Map(entries) => {
                write!(f, "{{")?;
                for (i, (k, v)) in entries.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{k} => {v}")?;
                }
                write!(f, "}}")
            }
        }
    }
}

// ── Environment ───────────────────────────────────────────────────────────────
//
// We use a simple stack of frames. Each frame corresponds to one Binder level.
// ExprKind::Var(Var::Bound(debruijn, breft)) indexes into this stack:
//   - debruijn=INNERMOST (0) → top frame
//   - breft.var.index()     → position within that frame

#[derive(Clone)]
pub struct Env {
    /// Outermost frame first, innermost (most recently pushed) last.
    bound: Vec<Vec<Value>>,
    /// Free/named variables (spec params, `Local`s, etc.)
    free: std::collections::HashMap<String, Value>,
}

impl Env {
    pub fn new() -> Self {
        Env { bound: vec![], free: std::collections::HashMap::new() }
    }

    pub fn insert_free(&mut self, name: impl Into<String>, val: Value) {
        self.free.insert(name.into(), val);
    }

    /// Push a new binder frame with the given argument values.
    fn push_frame(&self, args: Vec<Value>) -> Env {
        let mut child = Env { bound: self.bound.clone(), free: self.free.clone() };
        child.bound.push(args);
        child
    }

    fn lookup_bound(&self, debruijn: u32, var_idx: usize) -> Value {
        // debruijn=0 is INNERMOST = last pushed = last element of self.bound
        let frame_idx = self.bound.len().wrapping_sub(1 + debruijn as usize);
        self.bound
            .get(frame_idx)
            .and_then(|frame| frame.get(var_idx))
            .cloned()
            .unwrap_or_else(|| Value::Opaque(format!("unbound[{debruijn},{var_idx}]")))
    }
}

// ── Interpreter ───────────────────────────────────────────────────────────────

pub struct Interpreter;

impl Interpreter {
    pub fn eval(expr: &Expr, env: &Env) -> Value {
        Self::eval_kind(expr.kind(), env)
    }

    fn eval_kind(kind: &ExprKind, env: &Env) -> Value {
        match kind {
            // ── Leaves ────────────────────────────────────────────────────
            ExprKind::Constant(c) => Self::eval_constant(c),

            ExprKind::Var(var) => Self::eval_var(var, env),

            ExprKind::Local(local) => {
                let key = format!("{local:?}");
                env.free
                    .get(&key)
                    .cloned()
                    .unwrap_or_else(|| Value::Opaque(format!("local({key})")))
            }

            ExprKind::ConstDefId(def_id) => Value::Opaque(format!("const_def_id({def_id:?})")),

            // ── Structural ────────────────────────────────────────────────
            ExprKind::Tuple(flds) => {
                Value::Tuple(flds.iter().map(|e| Self::eval(e, env)).collect())
            }

            ExprKind::Ctor(ctor, flds) => {
                let name = format!("{:?}", ctor.def_id());
                let variant = ctor.variant_idx().index();
                let fields = flds.iter().map(|e| Self::eval(e, env)).collect();
                Value::Adt(name, variant, fields)
            }

            ExprKind::IsCtor(def_id, variant_idx, expr) => {
                match Self::eval(expr, env) {
                    Value::Adt(name, v, _) => {
                        Value::Bool(name == format!("{def_id:?}") && v == variant_idx.index())
                    }
                    _ => Value::Opaque("is_ctor-on-non-adt".into()),
                }
            }

            // ── Projections ───────────────────────────────────────────────
            ExprKind::FieldProj(inner, proj) => {
                let field = proj.field_idx() as usize;
                match Self::eval(inner, env) {
                    Value::Tuple(vs) | Value::Adt(_, _, vs) => {
                        vs.into_iter()
                            .nth(field)
                            .unwrap_or_else(|| Value::Opaque(format!("bad-field-{field}")))
                    }
                    other => Value::Opaque(format!("field-proj({other})")),
                }
            }

            ExprKind::PathProj(base, field_idx) => {
                let idx = u32::from(*field_idx) as usize;
                match Self::eval(base, env) {
                    Value::Tuple(vs) | Value::Adt(_, _, vs) => {
                        vs.into_iter()
                            .nth(idx)
                            .unwrap_or_else(|| Value::Opaque(format!("bad-path-field-{idx}")))
                    }
                    other => Value::Opaque(format!("path-proj({other})")),
                }
            }

            // ── Application ───────────────────────────────────────────────
            //
            // App is NOT curried in this AST. All args arrive at once.
            // We pattern-match on the function position directly before
            // evaluating it, since GlobalFunc/InternalFunc have no standalone value.
            ExprKind::App(func, _sort_args, args) => {
                let avals: Vec<Value> = args.iter().map(|a| Self::eval(a, env)).collect();
                Self::apply_func(func, avals, env)
            }

            // ── Lambda ────────────────────────────────────────────────────
            ExprKind::Abs(lam) => Value::Abs(lam.clone()),

            // ── Let binding ───────────────────────────────────────────────
            //
            // `let x = init in body`  — body is a Binder<Expr> with one bound var.
            ExprKind::Let(init, body) => {
                let init_val = Self::eval(init, env);
                // The binder introduces exactly one variable.
                let child_env = env.push_frame(vec![init_val]);
                Self::eval(body.skip_binder_ref(), &child_env)
            }

            // ── Operators ─────────────────────────────────────────────────
            ExprKind::BinaryOp(op, l, r) => {
                let lv = Self::eval(l, env);
                let rv = Self::eval(r, env);
                Self::eval_binop(op, lv, rv)
            }

            ExprKind::UnaryOp(op, e) => {
                match (op, Self::eval(e, env)) {
                    (UnOp::Not, Value::Bool(b)) => Value::Bool(!b),
                    (UnOp::Neg, Value::Int(n)) => Value::Int(-n),
                    (op, v) => Value::Opaque(format!("{op:?}({v})")),
                }
            }

            // ── Control flow ──────────────────────────────────────────────
            ExprKind::IfThenElse(cond, then_, else_) => {
                match Self::eval(cond, env) {
                    Value::Bool(true) => Self::eval(then_, env),
                    Value::Bool(false) => Self::eval(else_, env),
                    other => Value::Opaque(format!("if({other})")),
                }
            }

            // ── Quantifiers ───────────────────────────────────────────────
            //
            // These would need an SMT solver for real evaluation.
            // We evaluate the body symbolically with an opaque witness.
            ExprKind::Exists(binder) => {
                let vars = binder.vars();
                let witnesses: Vec<Value> = vars
                    .iter()
                    .enumerate()
                    .map(|(i, _)| Value::Opaque(format!("∃[{i}]")))
                    .collect();
                let child_env = env.push_frame(witnesses);
                let body_val = Self::eval(binder.skip_binder_ref(), &child_env);
                Value::Opaque(format!("∃. {body_val}"))
            }

            ExprKind::ForAll(binder) => {
                let vars = binder.vars();
                let witnesses: Vec<Value> = vars
                    .iter()
                    .enumerate()
                    .map(|(i, _)| Value::Opaque(format!("∀[{i}]")))
                    .collect();
                let child_env = env.push_frame(witnesses);
                let body_val = Self::eval(binder.skip_binder_ref(), &child_env);
                Value::Opaque(format!("∀. {body_val}"))
            }

            ExprKind::BoundedQuant(kind, rng, binder) => {
                use flux_middle::fhir::QuantKind;
                let lo = rng.start as i128;
                let hi = rng.end as i128;
                // For small concrete ranges we can fully expand.
                if hi - lo <= 64 {
                    let results: Vec<Value> = (lo..hi)
                        .map(|i| {
                            let child_env = env.push_frame(vec![Value::Int(i)]);
                            Self::eval(binder.skip_binder_ref(), &child_env)
                        })
                        .collect();
                    // If all results are booleans, reduce logically.
                    if results.iter().all(|v| matches!(v, Value::Bool(_))) {
                        let bools: Vec<bool> = results
                            .into_iter()
                            .map(|v| matches!(v, Value::Bool(true)))
                            .collect();
                        let result = match kind {
                            QuantKind::Exists => bools.into_iter().any(|b| b),
                            QuantKind::Forall => bools.into_iter().all(|b| b),
                        };
                        return Value::Bool(result);
                    }
                }
                Value::Opaque(format!("{kind:?} {lo}..{hi} <body>"))
            }

            // ── Unsupported / inference-only nodes ────────────────────────
            ExprKind::KVar(kvar) => Value::Opaque(format!("{kvar:?}")),
            ExprKind::Hole(_) => Value::Opaque("?hole".into()),
            ExprKind::Alias(alias, args) => Value::Opaque(format!("alias({alias:?})")),
            ExprKind::GlobalFunc(kind) => Value::Opaque(format!("func({kind:?})")),
            ExprKind::InternalFunc(kind) => Value::Opaque(format!("internal({kind:?})")),
        }
    }

    // ── Function application ──────────────────────────────────────────────────

    fn apply_func(func: &Expr, args: Vec<Value>, env: &Env) -> Value {
        match func.kind() {
            // Theory functions (map_store, map_get, set ops, etc.)
            ExprKind::GlobalFunc(SpecFuncKind::Thy(thy)) => Self::apply_thy_func(*thy, args),

            // User-defined spec functions — we can't reduce without their bodies,
            // but we record the call symbolically.
            ExprKind::GlobalFunc(SpecFuncKind::Def(did)) => {
                Value::Opaque(format!("{}({:?})", did.name(), args))
            }

            // Internal UIFs: Val(op), Rel(op), Cast, PtrSize
            ExprKind::InternalFunc(internal) => Self::apply_internal(internal, args),

            // Immediate lambda application — the only case the compiler allows
            // in non-index position (see the ExprKind::Abs doc comment).
            ExprKind::Abs(lam) => {
                // Lambda::apply takes &[Expr] and does the de Bruijn substitution at
                // the Expr level, giving back a fully substituted Expr we can eval.
                // Since we have Values not Exprs, we push a frame directly.
                let child_env = env.push_frame(args);
                // Use vars() which is public, body is private — evaluate via apply()
                // with dummy exprs just to get the binder structure, then re-eval.
                // Simplest correct approach: evaluate using the frame we just pushed.
                let substituted = lam.apply(
                    &(0..lam.vars().len())
                        .map(|i| {
                            Expr::bvar(
                                rustc_type_ir::INNERMOST,
                                rustc_type_ir::BoundVar::from_usize(i),
                                flux_middle::rty::BoundReftKind::Anon,
                            )
                        })
                        .collect::<Vec<_>>(),
                );
                Self::eval(&substituted, &child_env)
            }

            ExprKind::Ctor(ctor, _) => {
                let name = format!("{:?}", ctor.def_id());
                let variant = ctor.variant_idx().index();
                Value::Adt(name, variant, args)
            }

            // Nested App or anything else — evaluate the function position first,
            // then dispatch on its value.
            _ => {
                match Self::eval(func, env) {
                    Value::Abs(lam) => {
                        let child_env = env.push_frame(args.clone());
                        let placeholder_args: Vec<Expr> = (0..args.len())
                            .map(|i| {
                                Expr::bvar(
                                    rustc_type_ir::INNERMOST,
                                    rustc_type_ir::BoundVar::from_usize(i),
                                    flux_middle::rty::BoundReftKind::Anon,
                                )
                            })
                            .collect();
                        let substituted = lam.apply(&placeholder_args);
                        Self::eval(&substituted, &child_env)
                    }
                    other => Value::Opaque(format!("apply({other}, {args:?})")),
                }
            }
        }
    }

    // ── Theory function dispatch ───────────────────────────────────────────────

    fn apply_thy_func(thy: ThyFunc, mut args: Vec<Value>) -> Value {
        match thy {
            // ThyFunc::MapEmpty => Value::Map(vec![]),
            ThyFunc::MapStore if args.len() == 3 => {
                let val = args.pop().unwrap();
                let key = args.pop().unwrap();
                let map = args.pop().unwrap();
                Self::map_store(map, key, val)
            }
            ThyFunc::MapSelect if args.len() == 2 => {
                let key = args.pop().unwrap();
                let map = args.pop().unwrap();
                Self::map_get(&map, &key)
            }
            ThyFunc::SetEmpty => Value::Map(vec![]),
            ThyFunc::SetSng if args.len() == 1 => {
                Value::Map(vec![(args.pop().unwrap(), Value::Bool(true))])
            }
            ThyFunc::SetCup if args.len() == 2 => {
                let b = args.pop().unwrap();
                let a = args.pop().unwrap();
                Self::set_union(a, b)
            }
            ThyFunc::SetMem if args.len() == 2 => {
                let set = args.pop().unwrap();
                let elem = args.pop().unwrap();
                Value::Bool(Self::map_get(&set, &elem) == Value::Bool(true))
            }
            _ => Value::Opaque(format!("{thy:?}({args:?})")),
        }
    }

    // ── Internal UIF dispatch ─────────────────────────────────────────────────

    fn apply_internal(kind: &InternalFuncKind, args: Vec<Value>) -> Value {
        match kind {
            // Val(op): the "value" of a primop — evaluate it concretely if possible
            InternalFuncKind::Val(op) if args.len() == 2 => {
                Self::eval_binop(op, args[0].clone(), args[1].clone())
            }
            // Rel(op): the "relation" of a primop — same concrete evaluation
            InternalFuncKind::Rel(op) if args.len() == 2 => {
                Self::eval_binop(op, args[0].clone(), args[1].clone())
            }
            InternalFuncKind::Cast => {
                // Best effort: identity cast
                args.into_iter()
                    .next()
                    .unwrap_or(Value::Opaque("cast-no-arg".into()))
            }
            InternalFuncKind::PtrSize => Value::Int(std::mem::size_of::<*const ()>() as i128),
            _ => Value::Opaque(format!("{kind:?}({args:?})")),
        }
    }

    // ── Variable lookup ───────────────────────────────────────────────────────

    fn eval_var(var: &Var, env: &Env) -> Value {
        use rustc_type_ir::INNERMOST;
        match var {
            Var::Bound(debruijn, breft) => {
                let result = env.lookup_bound(debruijn.as_u32(), breft.var.index());
                eprintln!(
                    "lookup debruijn={} var={} → {:?}",
                    debruijn.as_u32(),
                    breft.var.index(),
                    result
                );
                result
            }
            Var::Free(name) => {
                env.free
                    .get(&format!("{name:?}"))
                    .cloned()
                    .unwrap_or_else(|| Value::Opaque(format!("free({name:?})")))
            }
            Var::EarlyParam(p) => {
                env.free
                    .get(&p.name.to_string())
                    .cloned()
                    .unwrap_or_else(|| Value::Opaque(format!("early({:?})", p.name)))
            }
            Var::EVar(evid) => Value::Opaque(format!("evar({evid:?})")),
            Var::ConstGeneric(param) => Value::Opaque(format!("const_generic({:?})", param.name)),
        }
    }

    // ── Constants ─────────────────────────────────────────────────────────────

    fn eval_constant(c: &Constant) -> Value {
        match c {
            Constant::Bool(b) => Value::Bool(*b),
            Constant::Int(n) => {
                // BigInt doesn't directly convert to i128; format and re-parse as fallback
                let s = format!("{n:?}");
                Value::Int(s.parse::<i128>().unwrap_or(0))
            }
            Constant::Str(sym) => Value::Str(sym.to_string()),
            Constant::Char(ch) => Value::Char(*ch),
            Constant::Real(r) => Value::Opaque(format!("{}.0", r.0)),
            Constant::BitVec(v, sz) => Value::Opaque(format!("bv({v}, {sz})")),
        }
    }

    // ── Binary operators ──────────────────────────────────────────────────────

    fn eval_binop(op: &BinOp, l: Value, r: Value) -> Value {
        match (op, &l, &r) {
            (BinOp::Eq, _, _) => Value::Bool(l == r),
            (BinOp::Ne, _, _) => Value::Bool(l != r),
            (BinOp::And, Value::Bool(a), Value::Bool(b)) => Value::Bool(*a && *b),
            (BinOp::Or, Value::Bool(a), Value::Bool(b)) => Value::Bool(*a || *b),
            (BinOp::Iff, Value::Bool(a), Value::Bool(b)) => Value::Bool(a == b),
            (BinOp::Imp, Value::Bool(a), Value::Bool(b)) => Value::Bool(!a || *b),
            (BinOp::Add(_), Value::Int(a), Value::Int(b)) => Value::Int(a + b),
            (BinOp::Sub(_), Value::Int(a), Value::Int(b)) => Value::Int(a - b),
            (BinOp::Mul(_), Value::Int(a), Value::Int(b)) => Value::Int(a * b),
            (BinOp::Div(_), Value::Int(a), Value::Int(b)) => {
                if *b == 0 {
                    Value::Opaque("div/0".into())
                } else {
                    Value::Int(a / b)
                }
            }
            (BinOp::Mod(_), Value::Int(a), Value::Int(b)) => Value::Int(a % b),
            (BinOp::Gt(_), Value::Int(a), Value::Int(b)) => Value::Bool(a > b),
            (BinOp::Ge(_), Value::Int(a), Value::Int(b)) => Value::Bool(a >= b),
            (BinOp::Lt(_), Value::Int(a), Value::Int(b)) => Value::Bool(a < b),
            (BinOp::Le(_), Value::Int(a), Value::Int(b)) => Value::Bool(a <= b),
            (BinOp::BitAnd(_), Value::Int(a), Value::Int(b)) => Value::Int(a & b),
            (BinOp::BitOr(_), Value::Int(a), Value::Int(b)) => Value::Int(a | b),
            (BinOp::BitXor(_), Value::Int(a), Value::Int(b)) => Value::Int(a ^ b),
            (BinOp::BitShl(_), Value::Int(a), Value::Int(b)) => Value::Int(a << b),
            (BinOp::BitShr(_), Value::Int(a), Value::Int(b)) => Value::Int(a >> b),
            _ => Value::Opaque(format!("{op:?}({l}, {r})")),
        }
    }

    // ── Map / Set helpers ─────────────────────────────────────────────────────

    fn map_store(map: Value, key: Value, value: Value) -> Value {
        let mut entries = match map {
            Value::Map(e) => e,
            _ => vec![],
        };
        if let Some(slot) = entries.iter_mut().find(|(k, _)| k == &key) {
            slot.1 = value;
        } else {
            entries.push((key, value));
        }
        Value::Map(entries)
    }

    fn map_get(map: &Value, key: &Value) -> Value {
        match map {
            Value::Map(entries) => {
                entries
                    .iter()
                    .find(|(k, _)| k == key)
                    .map(|(_, v)| v.clone())
                    .unwrap_or(Value::Opaque("map-miss".into()))
            }
            _ => Value::Opaque("map-get-on-non-map".into()),
        }
    }

    fn set_union(a: Value, b: Value) -> Value {
        match (a, b) {
            (Value::Map(mut ma), Value::Map(mb)) => {
                for (k, v) in mb {
                    if !ma.iter().any(|(ek, _)| ek == &k) {
                        ma.push((k, v));
                    }
                }
                Value::Map(ma)
            }
            (a, b) => Value::Opaque(format!("set_union({a}, {b})")),
        }
    }
}

pub struct Simplifier;

impl Simplifier {
    pub fn what_is(binder: &Binder<Expr>) -> std::collections::HashMap<String, Value> {
        // Step 1: substitute all bound vars with fresh named free variables
        let fresh_names: Vec<Name> = (0..binder.vars().len())
            .map(|i| Name::from_usize(i))
            .collect();
        let placeholders: Vec<Expr> = fresh_names.iter().map(|n| Expr::fvar(*n)).collect();
        let inner = binder.replace_bound_refts(&placeholders);

        // Step 2: walk looking for equalities, build solution map
        let mut solutions = std::collections::HashMap::new();
        let mut env = Env::new();
        for (i, name) in fresh_names.iter().enumerate() {
            env.insert_free(format!("{name:?}"), Value::Opaque(format!("b{i}")));
        }
        Self::collect_equalities(&inner, &env, &fresh_names, &mut solutions);
        solutions
    }

    fn collect_equalities(
        expr: &Expr,
        env: &Env,
        bound_names: &[Name],
        solutions: &mut std::collections::HashMap<String, Value>,
    ) {
        match expr.kind() {
            ExprKind::BinaryOp(BinOp::And, l, r) => {
                Self::collect_equalities(l, env, bound_names, solutions);
                Self::collect_equalities(r, env, bound_names, solutions);
            }
            ExprKind::BinaryOp(BinOp::Eq, lhs, rhs) => {
                // Check if either side is one of our placeholders
                let is_placeholder = |e: &Expr| -> Option<(usize, &str)> {
                    if let ExprKind::Var(Var::Free(name)) = e.kind() {
                        bound_names
                            .iter()
                            .enumerate()
                            .find(|(_, n)| *n == name)
                            .map(|(i, _)| (i, ""))
                    } else {
                        None
                    }
                };
                if let Some((i, _)) = is_placeholder(lhs) {
                    let val = Interpreter::eval(rhs, env);
                    solutions.insert(format!("b{i}"), val);
                } else if let Some((i, _)) = is_placeholder(rhs) {
                    let val = Interpreter::eval(lhs, env);
                    solutions.insert(format!("b{i}"), val);
                }
            }
            ExprKind::Exists(inner) | ExprKind::ForAll(inner) => {
                // Substitute inner bound vars with fresh names too, then recurse
                let inner_names: Vec<Name> = (bound_names.len()
                    ..bound_names.len() + inner.vars().len())
                    .map(|i| Name::from_usize(i))
                    .collect();
                let inner_placeholders: Vec<Expr> =
                    inner_names.iter().map(|n| Expr::fvar(*n)).collect();
                let inner_expr = inner.replace_bound_refts(&inner_placeholders);
                let mut inner_env: Env = env.clone();
                for (i, name) in inner_names.iter().enumerate() {
                    inner_env.insert_free(
                        format!("{name:?}"),
                        Value::Adt("_".into(), 0, (0..16).map(|_| Value::Map(vec![])).collect()),
                    );
                }
                let all_names: Vec<Name> = bound_names
                    .iter()
                    .chain(inner_names.iter())
                    .cloned()
                    .collect();
                Self::collect_equalities(&inner_expr, &inner_env, &all_names, solutions);
            }
            _ => {}
        }
    }
}
