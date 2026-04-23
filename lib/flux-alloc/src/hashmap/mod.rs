#![allow(dead_code)]
#![flux::defs {
    fn map_set<K, V>(m:Map<K, V>, k: K, v: V) -> Map<K, V> { map_store(m, k, v) }
    fn map_get<K, V>(m: Map<K, V>, k:K) -> V { map_select(m, k) }
    fn map_def<K, V>(v:V) -> Map<K, V> { map_default(v) }
}]

use std::{
    collections::HashMap,
    hash::{BuildHasher, Hash, RandomState},
};

use flux_attrs::*;
/// define a type indexed by a map
#[refined_by(vals: Map<K, V>)]
#[extern_spec(std::collections)]
struct HashMap<K, V, S = RandomState>;

// #[extern_spec]
// impl<K, V, S> HashMap<K, V, S> {
//     #[sig(fn(self: &strg HashMap<K,V, S>[@m], k: K, v: V) -> Option<V> ensures self: HashMap<K,V, S>[map_set(m.vals, k, v)])]
//     fn insert(&mut self, k: K, v: V) -> Option<V>
//     where
//         K: Eq + Hash,
//         S: BuildHasher;
// }
