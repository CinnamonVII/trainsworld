//! SoA buffers and transactional staging for WebGPU execution of `mode_choice_part`.
//!
//! Splits mode-choice inputs into three lifetimes so static pack/network geometry and
//! round-invariant period visit tables stay resident on GPU across crowding rounds:
//! - `GpuNetworkStage`: built once per `set_network`, independent of period & skims.
//! - `GpuPeriodStage`: built once per `(period q, part, chunk settings)`, reused across
//!   all crowding rounds `1..R` of that period.
//! - `StagedGpuPass`: per-pass transaction holding mutable solver state (`Period` / `Graph`)
//!   and skim inputs (`sk.r`, packed `sk.walk`, `sk.walk_max`) until `gpu_commit_pass` or
//!   `gpu_abort_pass` on WebGPU fallback.

use super::api::{od_sym_factors, put_fares, DemandApi, Period, SubOut};
use super::kernel::{
    assign_with_transfers, build_graph, dist_band, fast_exp, msa, round_costs, set_crowding,
    settled, walked, with_skims, xfer_index, FareSense, Gravity, Loads, NoRail, Params, Subzones,
    Zones, CHUNK_BLOCK, FARE_D_MIN, KE, NO_WALK, N_BANDS,
};
use super::lengths::{len_kernel, LenOut, N_LEN};
use super::network::PERIODS;

/// Number of origin subzones processed together per GPU batch (`ab_row`, `ab_y`, `pair_g`).
pub const GPU_U_BATCH: usize = 64;

/// Number of `u32` header offsets/counts in `GpuNetworkStage::meta()`.
pub const NET_META_LEN: usize = 38;

/// Offsets & dimensions packed into `net_meta` (shared between Rust, TS, and WGSL uniforms).
#[derive(Clone, Debug)]
pub struct DZoneMeta {
    pub key: i32,
    pub reg: u8,
    pub zid: u32,
    pub min: f32,
    pub max: f32,
    pub dv: f32,
    pub q0: u32,
    pub q1: u32,
    pub gx: i32,
    pub gy: i32,
    pub hx: i32,
    pub hy: i32,
    pub sup: u16,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PeriodStageKey {
    pub q: usize,
    pub part: (u32, u32),
    pub period_share_bits: u32,
    pub commute_chunk_bits: u32,
    pub chunk_salt: u32,
    pub prune_trips_bits: u32,
    pub net_epoch: u64,
}

pub struct GpuNetworkStage {
    #[allow(dead_code)]
    pub net_epoch: u64,
    pub s: usize,
    pub nb: usize,
    pub nq: usize,
    pub nu: usize,
    pub n_dz: usize,
    pub n_oz: usize,
    pub na_entries: usize,
    pub n_batches: usize,
    pub parted: bool,
    pub lk_zone_km: f32,

    /// Packed SoA `f32` storage buffer contents (`binding(1)`).
    pub net_f32: Vec<f32>,
    /// Packed SoA `u32` storage buffer contents (`binding(2)`).
    pub net_u32: Vec<u32>,
    /// Header words: `[s, nb, nq, nu, n_dz, n_oz, na_entries, n_batches, parted, u_batch, ...offsets]`
    pub meta: [u32; NET_META_LEN],

    // CPU-side metadata reused by `GpuPeriodStage` and `gpu_commit_pass`
    pub ozones: Vec<usize>,
    pub oz_u_range: Vec<(u32, u32)>,
    pub u_orig: Vec<u32>,
    pub dd_sub: Vec<u32>,
    pub dz: Vec<DZoneMeta>,
    pub dorder: Vec<(u32, u32)>,
    pub fw: Vec<[f32; 2]>,
    pub eg_k_flat: Vec<u32>,
    pub eg_n_flat: Vec<u8>,
    #[allow(dead_code)]
    pub acc_band: Vec<u32>,
    /// Per active origin ordinal `u_ord`, `(wmax_walk, wmax_ride, rides)` computed for current `base.pair_walk_max`?
    /// Note: `m_u` and `ride_b` are stored cleanly so `GpuPeriodStage` uses `p.pair_walk_max` dynamically.
    pub u_m_u: Vec<f32>,
    pub u_ride_b: Vec<f32>,
    pub u_has_rides: Vec<bool>,
    pub max_acc_span: usize,
}

pub struct GpuPeriodStage {
    pub key: PeriodStageKey,
    pub n_visits: usize,
    pub visit_f32: Vec<f32>,
    pub visit_u32: Vec<u32>,
    pub oz_visit_off: Vec<u32>,
    pub ou_period: Vec<f32>,

    // Scalar diagnostics/totals independent of skims (`sk`)
    pub trips_access: f64,
    #[allow(dead_code)]
    pub trips_total: f64,
    pub zone_pairs: u64,
    pub pairs: u64,
    pub pruned_trips: f64,
}

pub struct CrowdBackup {
    pub fac: Vec<f32>,
    pub left: Vec<f32>,
    pub uniform: bool,
    pub seen: Vec<f32>,
    pub ms: f64,
}

pub struct StagedGpuPass {
    pub q: usize,
    pub is_crowd: bool,
    pub t0: f64,
    pub period_idx: usize,
    pub period: Period,
    pub walk0: f64,
    pub backup: Option<CrowdBackup>,

    /// Per-pass dynamic skim buffers uploaded to GPU
    pub sk_r: Vec<f32>,
    /// Packed `u16` pairs (`(w0 as u32) | ((w1 as u32) << 16)`), `ceil(s*s / 2)` words
    pub sk_walk_u32: Vec<u32>,
    pub sk_walk_len: usize,
    pub sk_walk_max: Vec<f32>,
}

impl GpuNetworkStage {
    pub fn build(
        net_epoch: u64,
        z: &Zones,
        gr: &Gravity,
        sz: &Subzones,
        s: usize,
        p: &Params,
    ) -> Self {
        let nb = p.n_bands();
        let bands: Vec<(f32, f32)> = (0..nb).map(|b| p.band(b)).collect();
        let na_entries = sz.acc_st.len();
        let parted = !sz.part.is_empty();

        // Access entry walking distance and distance band (computed unconditionally so dynamic
        // `p.walk_rule` changes never require rebuilding `GpuNetworkStage`).
        let mut acc_walk = vec![0.0f32; na_entries];
        let mut acc_band = vec![0u32; na_entries];
        for u in 0..sz.n {
            for k in sz.acc_off[u] as usize..sz.acc_off[u + 1] as usize {
                acc_walk[k] = walked(sz, p, u, k);
                acc_band[k] = dist_band(sz.acc_d[k]) as u32;
            }
        }

        // Per-subzone car column factor `dv` & local street speed
        let dv: Vec<f32> = (0..sz.n)
            .map(|v| {
                let zj = sz.zone[v] as usize;
                if z.jobs[zj] > 0.0 {
                    (gr.col[zj] * sz.jobs[v] as f64 / z.jobs[zj]) as f32
                } else {
                    0.0
                }
            })
            .collect();
        let park: Vec<(f32, f32, f32)> = (0..z.n).map(|j| p.park_parts(z, j)).collect();
        let dens = z.density();
        let local: Vec<(f32, f32)> = (0..sz.n)
            .map(|v| p.place_min_per_km(dens[sz.zone[v] as usize]))
            .collect();

        // Station-choice offsets & exponential weights per entry
        let mut acc_m = vec![f32::INFINITY; sz.n];
        let mut ride_m = vec![f32::INFINITY; sz.n];
        let mut ride_b = vec![f32::INFINITY; sz.n];
        let mut acc_e = vec![0.0f32; nb * na_entries];
        let min_of = |r: std::ops::Range<usize>| {
            sz.acc_w[r].iter().cloned().fold(f32::INFINITY, f32::min)
        };
        for v in 0..sz.n {
            let m = min_of(sz.walk(v));
            let mr = m.min(min_of(sz.ride(v)));
            acc_m[v] = m;
            ride_m[v] = mr;
            ride_b[v] = sz.bound_min(p, v);
            for (b, &(th, _)) in bands.iter().enumerate() {
                for k in sz.walk(v) {
                    acc_e[b * na_entries + k] = fast_exp(-th * (sz.acc_w[k] - m));
                }
                for k in sz.ride(v) {
                    acc_e[b * na_entries + k] = fast_exp(-th * (sz.acc_w[k] - mr));
                }
            }
        }

        // Origin zones and ordinal origin subzone ordering
        let ozones: Vec<usize> = (0..z.n)
            .filter(|&i| sz.zo_off[i + 1] > sz.zo_off[i])
            .collect();
        let n_oz = ozones.len();
        let mut oz_u_range = Vec::with_capacity(n_oz);
        let mut u_orig = Vec::with_capacity(sz.zo.len());
        let mut u_oz_idx = Vec::with_capacity(sz.zo.len());
        for (oi, &zi) in ozones.iter().enumerate() {
            let u0 = u_orig.len() as u32;
            for &u in &sz.zo[sz.zo_off[zi] as usize..sz.zo_off[zi + 1] as usize] {
                u_orig.push(u);
                u_oz_idx.push(oi as u32);
            }
            let u1 = u_orig.len() as u32;
            oz_u_range.push((u0, u1));
        }
        let nu = u_orig.len();
        let n_batches = nu.div_ceil(GPU_U_BATCH);

        // Destination zones sorted identically to `mode.rs:391-469`
        let mut dzones: Vec<usize> = (0..z.n)
            .filter(|&j| sz.zd_off[j + 1] > sz.zd_off[j])
            .collect();
        let zmin = |j: usize| acc_m[sz.zd[sz.zd_off[j] as usize] as usize];
        dzones.sort_by_key(|&j| ((zmin(j) * 0.1) as u32, j));
        let n_dz = dzones.len();
        let nq = sz.zd.len();

        let mut dz = Vec::with_capacity(n_dz);
        let mut dd_f32 = Vec::with_capacity(nq * 12);
        let mut dd_sub = Vec::with_capacity(nq);
        let mut dd_dz_k = Vec::with_capacity(nq);
        let mut dd_part = Vec::with_capacity(nq);
        let mut eg_st_u32 = Vec::with_capacity(nq * KE);
        let mut eg_k_flat = Vec::with_capacity(nq * KE);
        let mut eg_n_flat = Vec::with_capacity(nq);
        let mut eg_e = vec![0.0f32; nb * nq * KE];
        let mut dz_q_range = Vec::with_capacity(n_dz * 2);

        // Inverted station -> egress list for `build_ab_y`: counts per station first
        let mut st_eg_cnt = vec![0u32; s];
        let mut total_walk_eg = 0usize;

        for (zk, &j) in dzones.iter().enumerate() {
            let q0 = dd_sub.len() as u32;
            let mut dvz = 0.0f32;
            for &v in &sz.zd[sz.zd_off[j] as usize..sz.zd_off[j + 1] as usize] {
                let v = v as usize;
                let q = dd_sub.len();
                let mut st = [s as u32; KE];
                let mut kk = [0u32; KE];
                let wl = sz.walk(v).len();
                for (n, k) in sz.walk(v).enumerate() {
                    st[n] = sz.acc_st[k];
                    kk[n] = k as u32;
                    st_eg_cnt[sz.acc_st[k] as usize] += 1;
                    total_walk_eg += 1;
                }
                for b in 0..nb {
                    let base = (b * nq + q) * KE;
                    for (n, k) in sz.walk(v).enumerate() {
                        eg_e[base + n] = acc_e[b * na_entries + k];
                    }
                }
                let eg_far = sz.walk(v).map(|k| sz.acc_d[k]).fold(0.0, f32::max);
                eg_st_u32.extend_from_slice(&st);
                eg_k_flat.extend_from_slice(&kk);
                eg_n_flat.push(wl as u8);
                dvz += dv[v];
                let (fixed, paid, free) = park[j];
                dd_f32.extend_from_slice(&[
                    sz.cx[v], sz.cy[v], local[v].0, local[v].1,
                    fixed, dv[v], acc_m[v], paid,
                    free, eg_far, 0.0, 0.0,
                ]);
                dd_sub.push(v as u32);
                dd_dz_k.push(zk as u32);
                dd_part.push(if parted { sz.part[v] as u32 } else { 0 });
            }
            let q1 = dd_sub.len() as u32;
            dz_q_range.push(q0);
            dz_q_range.push(q1);
            dz.push(DZoneMeta {
                key: gr.key(z, j),
                reg: z.region[j],
                zid: j as u32,
                min: zmin(j),
                max: if q1 > q0 { dd_f32[(q1 as usize - 1) * 12 + 6] } else { f32::INFINITY },
                dv: dvz,
                q0,
                q1,
                gx: z.gx[j],
                gy: z.gy[j],
                hx: z.hx(j),
                hy: z.hy(j),
                sup: p.routes.as_ref().map_or(0, |r| r.sup_of(j)),
            });
        }

        // Build inverted CSR `st_eg_off`, `st_eg_items` (`[q, kk]`)
        let mut st_eg_off = vec![0u32; s + 1];
        for st_i in 0..s {
            st_eg_off[st_i + 1] = st_eg_off[st_i] + st_eg_cnt[st_i];
        }
        let mut st_eg_cur = st_eg_off[..s].to_vec();
        let mut st_eg_items = vec![0u32; total_walk_eg * 2];
        for q in 0..nq {
            let wn = eg_n_flat[q] as usize;
            for kk in 0..wn {
                let st_i = eg_st_u32[q * KE + kk] as usize;
                let pos = st_eg_cur[st_i] as usize;
                st_eg_cur[st_i] += 1;
                st_eg_items[2 * pos] = q as u32;
                st_eg_items[2 * pos + 1] = kk as u32;
            }
        }

        // Chunk block ordering (`mode.rs:473-475`)
        let blk = |j: usize| {
            ((z.gx[j].div_euclid(CHUNK_BLOCK) as u32 & 0xffff) << 16)
                | (z.gy[j].div_euclid(CHUNK_BLOCK) as u32 & 0xffff)
        };
        let mut dorder: Vec<(u32, u32)> = dz
            .iter()
            .enumerate()
            .map(|(k, zj)| (blk(zj.zid as usize), k as u32))
            .collect();
        dorder.sort_by_key(|&(b, k)| (b, dz[k as usize].zid));
        let fw: Vec<[f32; 2]> = gr
            .ftab
            .iter()
            .enumerate()
            .map(|(k, &f)| [f, gr.walk_c(k)])
            .collect();

        // Origin metadata (`u_c_u`, `u_meta_f32`, `u_meta_u32`)
        let mut u_c_u = vec![0.0f32; nu * nb];
        let mut u_meta_f32 = Vec::with_capacity(nu * 8);
        let mut u_meta_u32 = Vec::with_capacity(nu * 8);
        let mut u_m_u = Vec::with_capacity(nu);
        let mut u_ride_b = Vec::with_capacity(nu);
        let mut u_has_rides = Vec::with_capacity(nu);
        let mut max_acc_span = 0usize;

        for (u_ord, &orig_u_u32) in u_orig.iter().enumerate() {
            let u = orig_u_u32 as usize;
            let walk = sz.walk(u);
            let ride = sz.ride(u);
            max_acc_span = max_acc_span.max(ride.end.saturating_sub(walk.start));
            let m_u = acc_m[u];
            let m_r = ride_m[u];
            let rb = ride_b[u];
            let rides = !ride.is_empty();
            u_m_u.push(m_u);
            u_ride_b.push(rb);
            u_has_rides.push(rides);
            for (b, &(th, _)) in bands.iter().enumerate() {
                u_c_u[u_ord * nb + b] = if walk.is_empty() {
                    0.0
                } else {
                    fast_exp(-th * (m_u - m_r))
                };
            }
            u_meta_f32.extend_from_slice(&[
                sz.cx[u], sz.cy[u], local[u].0, local[u].1,
                sz.home_park[u], m_u, m_r, rb,
            ]);
            u_meta_u32.extend_from_slice(&[
                orig_u_u32,
                u_oz_idx[u_ord],
                walk.start as u32,
                walk.end as u32,
                ride.end as u32,
                if parted { sz.part[u] as u32 } else { 0 },
                0,
                0,
            ]);
        }

        // Precompute per-batch active stations & access lists (`batch_meta_u32` etc.) so
        // `split_to_od_batch` and `slow_apply_batch` iterate stations strictly in ascending `u` order.
        let mut batch_meta_u32 = Vec::with_capacity(n_batches * 4);
        let mut batch_active_a_off = Vec::with_capacity(n_batches + 1);
        let mut batch_active_a = Vec::new();
        let mut batch_a_acc_off = Vec::new();
        let mut batch_acc_items = Vec::new();
        let mut st_seen = vec![u32::MAX; s];
        let mut st_list_tmp: Vec<(usize, Vec<u32>)> = Vec::new();

        batch_active_a_off.push(0);
        for b_idx in 0..n_batches {
            let u_start = b_idx * GPU_U_BATCH;
            let u_end = (u_start + GPU_U_BATCH).min(nu);
            let oz_first = u_oz_idx[u_start];
            let oz_last = u_oz_idx[u_end - 1];
            batch_meta_u32.extend_from_slice(&[u_start as u32, u_end as u32, oz_first, oz_last]);

            st_list_tmp.clear();
            for u_local in 0..(u_end - u_start) {
                let u_ord = u_start + u_local;
                let u = u_orig[u_ord] as usize;
                let ws = sz.acc_off[u] as usize;
                let we = sz.acc_mid[u] as usize;
                let re = sz.acc_off[u + 1] as usize;
                for k in ws..re {
                    let a = sz.acc_st[k] as usize;
                    let is_walk = (k < we) as u32;
                    let pos = if st_seen[a] != b_idx as u32 {
                        st_seen[a] = b_idx as u32;
                        let p = st_list_tmp.len();
                        st_list_tmp.push((a, Vec::new()));
                        p
                    } else {
                        st_list_tmp.iter().position(|(st_a, _)| *st_a == a).unwrap()
                    };
                    st_list_tmp[pos].1.push((u_local as u32) | (is_walk << 31));
                    st_list_tmp[pos].1.push(k as u32);
                }
            }
            // Canonical ascending station order
            st_list_tmp.sort_by_key(|(a, _)| *a);
            for (a, items) in &st_list_tmp {
                batch_a_acc_off.push((batch_acc_items.len() / 2) as u32);
                batch_active_a.push(*a as u32);
                batch_acc_items.extend_from_slice(items);
            }
            batch_active_a_off.push(batch_active_a.len() as u32);
        }
        batch_a_acc_off.push((batch_acc_items.len() / 2) as u32);

        // Length spread kernel (`LenKernel`) tables
        let lk = len_kernel(z.zone_m);
        let (lk_zone_km, lk_off, lk_ent) = lk.tables();
        let mut lk_ent_f32 = Vec::with_capacity(lk_ent.len() * 2);
        let mut lk_ent_bin = Vec::with_capacity(lk_ent.len());
        for &(bin, w, wk) in lk_ent {
            lk_ent_bin.push(bin as u32);
            lk_ent_f32.push(w);
            lk_ent_f32.push(wk);
        }

        // Pack `net_f32` and record offsets in `meta`
        let mut net_f32 = Vec::new();
        let mut meta = [0u32; NET_META_LEN];
        meta[0] = s as u32;
        meta[1] = nb as u32;
        meta[2] = nq as u32;
        meta[3] = nu as u32;
        meta[4] = n_dz as u32;
        meta[5] = n_oz as u32;
        meta[6] = na_entries as u32;
        meta[7] = n_batches as u32;
        meta[8] = parted as u32;
        meta[9] = GPU_U_BATCH as u32;

        // `f32` offsets: meta[10..19]
        meta[10] = net_f32.len() as u32; // off_bands
        for &(th, off) in &bands {
            net_f32.push(th);
            net_f32.push(off);
        }
        meta[11] = net_f32.len() as u32; // off_dd_f32
        net_f32.extend_from_slice(&dd_f32);
        meta[12] = net_f32.len() as u32; // off_eg_e
        net_f32.extend_from_slice(&eg_e);
        meta[13] = net_f32.len() as u32; // off_acc_e
        net_f32.extend_from_slice(&acc_e);
        meta[14] = net_f32.len() as u32; // off_acc_d
        net_f32.extend_from_slice(&sz.acc_d);
        meta[15] = net_f32.len() as u32; // off_acc_walk
        net_f32.extend_from_slice(&acc_walk);
        meta[16] = net_f32.len() as u32; // off_u_c_u
        net_f32.extend_from_slice(&u_c_u);
        meta[17] = net_f32.len() as u32; // off_u_meta_f32
        net_f32.extend_from_slice(&u_meta_f32);
        meta[18] = net_f32.len() as u32; // off_lk_ent_f32
        net_f32.extend_from_slice(&lk_ent_f32);
        if net_f32.is_empty() {
            net_f32.push(0.0);
        }

        // `u32` offsets: meta[19..36]
        let mut net_u32 = Vec::new();
        meta[19] = net_u32.len() as u32; // off_eg_st
        net_u32.extend_from_slice(&eg_st_u32);
        meta[20] = net_u32.len() as u32; // off_eg_k
        net_u32.extend_from_slice(&eg_k_flat);
        meta[21] = net_u32.len() as u32; // off_eg_n
        net_u32.extend(eg_n_flat.iter().map(|&v| v as u32));
        meta[22] = net_u32.len() as u32; // off_dd_dz_k
        net_u32.extend_from_slice(&dd_dz_k);
        meta[23] = net_u32.len() as u32; // off_dd_part
        net_u32.extend_from_slice(&dd_part);
        meta[24] = net_u32.len() as u32; // off_acc_st
        net_u32.extend_from_slice(&sz.acc_st);
        meta[25] = net_u32.len() as u32; // off_acc_band
        net_u32.extend_from_slice(&acc_band);
        meta[26] = net_u32.len() as u32; // off_st_eg_off
        net_u32.extend_from_slice(&st_eg_off);
        meta[27] = net_u32.len() as u32; // off_st_eg_items
        net_u32.extend_from_slice(&st_eg_items);
        meta[28] = net_u32.len() as u32; // off_u_meta_u32
        net_u32.extend_from_slice(&u_meta_u32);
        meta[29] = net_u32.len() as u32; // off_lk_off
        net_u32.extend_from_slice(lk_off);
        meta[30] = net_u32.len() as u32; // off_lk_ent_bin
        net_u32.extend_from_slice(&lk_ent_bin);
        meta[31] = net_u32.len() as u32; // off_batch_meta_u32
        net_u32.extend_from_slice(&batch_meta_u32);
        meta[32] = net_u32.len() as u32; // off_batch_active_a_off
        net_u32.extend_from_slice(&batch_active_a_off);
        meta[33] = net_u32.len() as u32; // off_batch_active_a
        net_u32.extend_from_slice(&batch_active_a);
        meta[34] = net_u32.len() as u32; // off_batch_a_acc_off
        net_u32.extend_from_slice(&batch_a_acc_off);
        meta[35] = net_u32.len() as u32; // off_batch_acc_items
        net_u32.extend_from_slice(&batch_acc_items);
        meta[36] = net_u32.len() as u32; // off_dz_q_range
        net_u32.extend_from_slice(&dz_q_range);
        meta[37] = max_acc_span as u32;
        if net_u32.is_empty() {
            net_u32.push(0);
        }

        GpuNetworkStage {
            net_epoch,
            s,
            nb,
            nq,
            nu,
            n_dz,
            n_oz,
            na_entries,
            n_batches,
            parted,
            lk_zone_km,
            net_f32,
            net_u32,
            meta,
            ozones,
            oz_u_range,
            u_orig,
            dd_sub,
            dz,
            dorder,
            fw,
            eg_k_flat,
            eg_n_flat,
            acc_band,
            u_m_u,
            u_ride_b,
            u_has_rides,
            max_acc_span,
        }
    }
}

impl GpuPeriodStage {
    pub fn build(
        key: PeriodStageKey,
        net_stage: &GpuNetworkStage,
        z: &Zones,
        gr: &Gravity,
        sz: &Subzones,
        nr: &NoRail,
        p: &Params,
    ) -> Self {
        let share = p.period_share as f64;
        let trips_total = gr.total * share;
        let mut ou_period = vec![0.0f32; net_stage.nu];
        let mut max_wmax_oz = vec![f32::NEG_INFINITY; net_stage.n_oz];

        let (part_k, part_n) = key.part;
        for (oi, &_i) in net_stage.ozones.iter().enumerate() {
            if part_n > 1 && oi as u32 % part_n != part_k {
                continue;
            }
            let (u0, u1) = net_stage.oz_u_range[oi];
            let mut mw = f32::NEG_INFINITY;
            for u_ord in u0 as usize..u1 as usize {
                let u = net_stage.u_orig[u_ord] as usize;
                ou_period[u_ord] = sz.origin_weight(u, z, gr, share);
                let lim = p.pair_walk_max
                    - if net_stage.u_has_rides[u_ord] {
                        net_stage.u_ride_b[u_ord]
                    } else {
                        net_stage.u_m_u[u_ord]
                    };
                mw = mw.max(lim);
            }
            max_wmax_oz[oi] = mw;
        }

        let n_dz = net_stage.n_dz;
        let dz = &net_stage.dz;
        let mut fz = vec![[0f32, 0.0, p.road_detour, 0.0]; n_dz];
        let mut fzw = vec![0f32; n_dz];
        let mut fzb = vec![0u8; n_dz];
        let mut fm = vec![1f32; n_dz];
        let mut run: Vec<(usize, f64)> = Vec::new();

        let pair_walk = gr.wroutes.is_some() || !gr.wf_off.is_empty();
        let drive = p.routes.as_deref().filter(|r| r.has_drive());
        let ride_min2 = (p.ride_min_km * 1000.0).powi(2);

        let mut oz_visit_off = Vec::with_capacity(net_stage.n_oz + 1);
        let mut visit_f32: Vec<f32> = Vec::new();
        let mut visit_u32: Vec<u32> = Vec::new();
        let mut trips_access = 0.0f64;
        let mut zone_pairs = 0u64;
        let mut pairs = 0u64;
        let mut pruned_trips = 0.0f64;

        let dd_f32_off = net_stage.meta[11] as usize;
        let dd_f32 = &net_stage.net_f32[dd_f32_off..];
        let dd_acc_m = |q: u32| dd_f32[q as usize * 12 + 6];

        // Compute `reach(oi)` using `max_wmax_oz[oi]` — `dstep` is monotonic 10-minute blocks.
        // We can scan `0..nzi` where `nzi` includes all dzones within `max_wmax_oz[oi]`.
        let zmin_dstep: Vec<u32> = dz.iter().map(|zj| (zj.min * 0.1) as u32).collect();

        oz_visit_off.push(0);
        for (oi, &i) in net_stage.ozones.iter().enumerate() {
            if part_n > 1 && oi as u32 % part_n != part_k {
                oz_visit_off.push((visit_u32.len() / 4) as u32);
                continue;
            }
            let (u0, u1) = net_stage.oz_u_range[oi];
            let base = gr.base(z, i);
            let mut ou_zone = 0.0f32;
            for u_ord in u0 as usize..u1 as usize {
                ou_zone += ou_period[u_ord];
            }
            let ri = z.region[i];
            let nzi = zmin_dstep.partition_point(|&m| m as f32 * 10.0 <= max_wmax_oz[oi]);
            let (gxi, gyi, hxi, hyi) = (z.gx[i], z.gy[i], z.hx(i), z.hy(i));
            for ((o, zb), zj) in fz[..nzi].iter_mut().zip(fzb.iter_mut()).zip(dz) {
                let k = (base + zj.key) as usize;
                let [mut f, w] = net_stage.fw[k];
                let (dx, dy) = (zj.gx - gxi, zj.gy - gyi);
                let (fx, fy) = ((zj.hx - hxi) as f32 * 0.5, (zj.hy - hyi) as f32 * 0.5);
                let d2 = (fx * fx + fy * fy) * z.zone_m * z.zone_m;
                let long = (d2 >= ride_min2) as u32 as f32;
                *zb = p.band_of_d2(d2) as u8;
                if zj.reg != ri {
                    f *= gr.border;
                }
                let mut w_eff = w;
                if pair_walk && w < NO_WALK {
                    w_eff = gr.walk_cost_at(i, zj.zid as usize, k);
                }
                let df = if let Some(r) = drive {
                    r.drive_factor_by(i, dx, dy, zj.reg, zj.sup) * p.road_routed_scale
                } else {
                    p.road_detour
                };
                *o = [f, w_eff, df, long];
            }
            for (wz, (o, zj)) in fzw[..nzi].iter_mut().zip(fz.iter().zip(dz)) {
                *wz = if o[1] < NO_WALK {
                    nr.split(gr, p, i, zj.zid as usize, (base + zj.key) as usize)
                } else {
                    0.0
                };
            }

            fm[..nzi].fill(1.0);
            if p.commute_chunk > 0.0 {
                let c = p.commute_chunk as f64;
                let mut g = 0;
                while g < net_stage.dorder.len() {
                    let b = net_stage.dorder[g].0;
                    run.clear();
                    while g < net_stage.dorder.len() && net_stage.dorder[g].0 == b {
                        let k = net_stage.dorder[g].1 as usize;
                        g += 1;
                        if k < nzi {
                            let t = ou_zone as f64 * dz[k].dv as f64 * fz[k][0] as f64;
                            if t > 0.0 && t < c {
                                run.push((k, t));
                            }
                        }
                    }
                    if run.is_empty() {
                        continue;
                    }
                    let total: f64 = run.iter().map(|e| e.1).sum();
                    let draws = (total / c).ceil();
                    if draws >= run.len() as f64 {
                        continue;
                    }
                    let step = total / draws;
                    let at = super::intercity::unit_hash(
                        i as u32 ^ p.chunk_salt.wrapping_mul(0x27d4eb2f),
                        b,
                    );
                    let before = |x: f64| (x / step - at).ceil().clamp(0., draws);
                    let mut lo = 0.0;
                    for &(k, t) in &run {
                        let hi = lo + t;
                        fm[k] = ((before(hi) - before(lo)) * step / t) as f32;
                        lo = hi;
                    }
                }
            }

            // Compute exact `trips_access`, `zone_pairs`, `pairs`, `pruned_trips` AND emit `oz_visits`
            let mut visited_flag = vec![false; nzi];
            for (ui, u_ord) in (u0 as usize..u1 as usize).enumerate() {
                let rides = net_stage.u_has_rides[u_ord];
                let wmax_u = p.pair_walk_max - net_stage.u_m_u[u_ord];
                let wmax_r = p.pair_walk_max - net_stage.u_ride_b[u_ord];
                let lim_u = if rides { wmax_r } else { wmax_u };
                let nz_u = zmin_dstep.partition_point(|&m| m as f32 * 10.0 <= lim_u);
                let ou_u = ou_period[u_ord];
                for zk in 0..nz_u {
                    let zj = &dz[zk];
                    let [f, _, _, long] = fz[zk];
                    let by_ride = rides && long > 0.0;
                    let wmax = if by_ride { wmax_r } else { wmax_u };
                    if zj.min > wmax {
                        continue;
                    }
                    if p.prune_trips > 0.0 && ou_zone * zj.dv * f < p.prune_trips {
                        pruned_trips += (ou_u * zj.dv * f) as f64;
                        continue;
                    }
                    let m = fm[zk];
                    if m == 0.0 {
                        continue;
                    }
                    if ui == 0 {
                        zone_pairs += 1;
                    }
                    trips_access += (ou_u * f * m * zj.dv) as f64;
                    let nv = if zj.max <= wmax {
                        zj.q1 - zj.q0
                    } else {
                        let lo = zj.q0;
                        let hi = zj.q1;
                        let mut l = 0u32;
                        let mut r = hi - lo;
                        while l < r {
                            let mid = l + (r - l) / 2;
                            if dd_acc_m(lo + mid) <= wmax {
                                l = mid + 1;
                            } else {
                                r = mid;
                            }
                        }
                        l
                    };
                    pairs += nv as u64;
                    visited_flag[zk] = true;
                }
            }

            let si = z.side(i);
            for zk in 0..nzi {
                if !visited_flag[zk] {
                    continue;
                }
                let zj = &dz[zk];
                let [f, cw, df, long] = fz[zk];
                let f_m = f * fm[zk];
                let w0 = fzw[zk];
                let zb = fzb[zk];
                // Compute `gr.walk_factor` only when straight distance between zones can satisfy `WALK_REACH_KM`
                let d_zone_approx =
                    ((zj.hx - hxi) as f32 * 0.5 * z.zone_m).hypot((zj.hy - hyi) as f32 * 0.5 * z.zone_m);
                let wf = if d_zone_approx <= super::kernel::WALK_REACH_KM * 1000.0 + 3.0 * z.zone_m {
                    gr.walk_factor(i, zj.zid as usize)
                } else {
                    1.0
                };
                visit_f32.extend_from_slice(&[f_m, cw, df, w0, wf, long, f32::from_bits(zb as u32), 0.0]);
                let hdx = (zj.hx - hxi) as i16 as u16 as u32;
                let hdy = (zj.hy - hyi) as i16 as u16 as u32;
                let sides = (si + z.side(zj.zid as usize) - 2) as u32;
                let long_bit = (long > 0.0) as u32;
                let meta_bits = (zb as u32) | (long_bit << 8) | (sides << 9);
                visit_u32.extend_from_slice(&[
                    zk as u32,
                    meta_bits,
                    oi as u32,
                    hdx | (hdy << 16),
                ]);
            }
            oz_visit_off.push((visit_u32.len() / 4) as u32);
        }

        let n_visits = visit_u32.len() / 4;
        if visit_f32.is_empty() {
            visit_f32.resize(8, 0.0);
        }
        if visit_u32.is_empty() {
            visit_u32.resize(4, 0);
        }
        if ou_period.is_empty() {
            ou_period.push(0.0);
        }

        GpuPeriodStage {
            key,
            n_visits,
            visit_f32,
            visit_u32,
            oz_visit_off,
            ou_period,
            trips_access,
            trips_total,
            zone_pairs,
            pairs,
            pruned_trips,
        }
    }
}

pub fn pack_sk_walk_u16(walk: &[u16]) -> Vec<u32> {
    let mut out = Vec::with_capacity(walk.len().div_ceil(2));
    for chunk in walk.chunks(2) {
        let lo = chunk[0] as u32;
        let hi = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        out.push(lo | (hi << 16));
    }
    if out.is_empty() {
        out.push(0);
    }
    out
}

/// Pack pass scalar uniforms into a flat `[f32]` array (uploaded as a single `uniform` buffer
/// in `DemandGpuContext`, mixing bitcast `u32` words cleanly).
pub fn build_pass_uniforms_f32(p: &Params, net_stage: &GpuNetworkStage, rule: bool) -> Vec<f32> {
    vec![
        f32::from_bits(net_stage.s as u32),                    // 0: s
        f32::from_bits(net_stage.nb as u32),                   // 1: nb
        f32::from_bits(net_stage.nq as u32),                   // 2: nq
        f32::from_bits(net_stage.nu as u32),                   // 3: nu
        f32::from_bits(net_stage.n_dz as u32),                 // 4: n_dz
        f32::from_bits(net_stage.n_oz as u32),                 // 5: n_oz
        f32::from_bits(net_stage.na_entries as u32),           // 6: na_entries
        f32::from_bits(net_stage.n_batches as u32),            // 7: n_batches
        f32::from_bits(net_stage.parted as u32),               // 8: parted
        f32::from_bits(rule as u32),                           // 9: rule
        f32::from_bits(GPU_U_BATCH as u32),                    // 10: u_batch
        p.pair_walk_max,                                       // 11: pair_walk_max
        p.fare_min,                                            // 12: fare_min
        p.car_own_km_min(),                                    // 13: car_own_km_min
        p.w_drive,                                             // 14: w_drive
        p.beta,                                                // 15: beta
        p.damp_min,                                            // 16: damp_min
        p.asc_rail,                                            // 17: asc_rail
        p.asc_walk(),                                          // 18: asc_walk
        p.ratio_fade_hi,                                       // 19: ratio_fade_hi
        1.0 / (p.ratio_fade_hi - p.ratio_fade_lo),             // 20: ratio_kf
        p.road_local_km,                                       // 21: road_local_km
        p.road_end_km,                                         // 22: road_end_km
        60.0 / p.road_open_kmh,                                // 23: road_open_min_per_km
        p.road_detour,                                         // 24: road_detour
        p.walk_rule,                                           // 25: walk_rule
        net_stage.lk_zone_km,                                  // 26: lk_zone_km
        f32::from_bits(net_stage.meta[10]),                    // 27+: net_f32 offsets (meta[10..19])
        f32::from_bits(net_stage.meta[11]),
        f32::from_bits(net_stage.meta[12]),
        f32::from_bits(net_stage.meta[13]),
        f32::from_bits(net_stage.meta[14]),
        f32::from_bits(net_stage.meta[15]),
        f32::from_bits(net_stage.meta[16]),
        f32::from_bits(net_stage.meta[17]),
        f32::from_bits(net_stage.meta[18]),
        f32::from_bits(net_stage.meta[19]),                    // 36+: net_u32 offsets (meta[19..36])
        f32::from_bits(net_stage.meta[20]),
        f32::from_bits(net_stage.meta[21]),
        f32::from_bits(net_stage.meta[22]),
        f32::from_bits(net_stage.meta[23]),
        f32::from_bits(net_stage.meta[24]),
        f32::from_bits(net_stage.meta[25]),
        f32::from_bits(net_stage.meta[26]),
        f32::from_bits(net_stage.meta[27]),
        f32::from_bits(net_stage.meta[28]),
        f32::from_bits(net_stage.meta[29]),
        f32::from_bits(net_stage.meta[30]),
        f32::from_bits(net_stage.meta[31]),
        f32::from_bits(net_stage.meta[32]),
        f32::from_bits(net_stage.meta[33]),
        f32::from_bits(net_stage.meta[34]),
        f32::from_bits(net_stage.meta[35]),
        f32::from_bits(net_stage.meta[36]),                    // 53: off_dz_q_range
        f32::from_bits(0),                                     // 54: n_visits (filled in JS)
        f32::from_bits(0),                                     // 55: off_ou_period (filled in JS)
        f32::from_bits(0),                                     // 56: off_oz_visit_off (filled in JS)
        f32::from_bits((net_stage.s * net_stage.s) as u32),    // 57: off_sk_walk_max
        f32::from_bits(0),                                     // 58: batch_idx
        f32::from_bits(0),                                     // 59: batch_v_start
        f32::from_bits(0),                                     // 60: batch_v_end
        f32::from_bits(0),                                     // 61: batch_active_a_lo
        f32::from_bits(0),                                     // 62: batch_active_a_hi
        f32::from_bits(0),                                     // 63: padding (64 floats = 256 bytes)
    ]
}

impl DemandApi {
    pub(crate) fn ensure_gpu_stages(&mut self, q: usize) -> Option<(usize, &GpuNetworkStage, &GpuPeriodStage)> {
        let sz = self.sz.as_ref()?;
        let s = self.svc.as_ref()?.st_x.len();
        if self.gpu_net.is_none() {
            self.gpu_net_epoch = self.gpu_net_epoch.wrapping_add(1);
            let st = GpuNetworkStage::build(
                self.gpu_net_epoch,
                &self.z,
                &self.gr,
                sz,
                s,
                &self.base,
            );
            self.gpu_net = Some(st);
            self.gpu_per.clear();
        }
        let p = self.params(q);
        let key = PeriodStageKey {
            q,
            part: self.part,
            period_share_bits: p.period_share.to_bits(),
            commute_chunk_bits: p.commute_chunk.to_bits(),
            chunk_salt: p.chunk_salt,
            prune_trips_bits: p.prune_trips.to_bits(),
            net_epoch: self.gpu_net_epoch,
        };
        let net_ref = self.gpu_net.as_ref().unwrap();
        let idx = if let Some(pos) = self.gpu_per.iter().position(|ps| ps.key == key) {
            pos
        } else {
            let ps = GpuPeriodStage::build(key, net_ref, &self.z, &self.gr, sz, &self.nr, &p);
            self.gpu_per.push(ps);
            self.gpu_per.len() - 1
        };
        let net_ref = self.gpu_net.as_ref().unwrap();
        let per_ref = &self.gpu_per[idx];
        Some((idx, net_ref, per_ref))
    }

    pub(crate) fn prepare_solve_gpu_inner(&mut self, q: usize) -> i32 {
        if self.staged_gpu.is_some() {
            self.abort_gpu_inner();
        }
        let t0 = super::clock::now_ms();
        if q >= PERIODS || self.svc.is_none() {
            return 0;
        }
        if self.sz.is_none() {
            // Empty subzone case completes synchronously without WebGPU
            return if self.solve(q) { 2 } else { 0 };
        }
        let (period_idx, net_ref, per_ref) = match self.ensure_gpu_stages(q) {
            Some(v) => v,
            None => return 0,
        };
        let s = net_ref.s;
        let max_acc_span = net_ref.max_acc_span;
        if per_ref.pairs < 32 || per_ref.n_visits < 64 {
            return if self.solve(q) { 2 } else { 0 };
        }
        let svc = self.svc.as_ref().unwrap();
        let p = self.params(q);
        let trips = self.gr.total * p.period_share as f64;
        let walk0 = self.walk_all * p.period_share as f64;
        let net = Self::period_net(svc, q);
        let mut g = build_graph(&net, &p);
        put_fares(
            &mut g,
            &self.fare_curve,
            &self.fare_km,
            &self.fare_mul,
            self.base.fare_min,
            self.fare_per_km,
            self.base.fare_w,
        );
        let fac = vec![1.0; g.rn_ride.len()];
        set_crowding(&mut g, &fac);
        let xi = xfer_index(&g);

        let (sk_r, sk_walk_u32, sk_walk_len, sk_walk_max) = with_skims(&g, &p, &self.want, |sk| {
            (
                sk.r.clone(),
                pack_sk_walk_u16(&sk.walk),
                sk.walk.len(),
                sk.walk_max.iter().map(|&v| v as f32).collect::<Vec<f32>>(),
            )
        });
        let rule = p.walk_rule > 0.0 && sk_walk_len == s * s && sk_walk_max.len() == s;
        if rule && max_acc_span > 64 {
            return if self.solve(q) { 2 } else { 0 };
        }
        let avg = Loads::zero(g.n_st, g.n - g.n_st);
        let xfer = vec![0.0; xi.len];
        let period = Period {
            net,
            p,
            g,
            avg,
            rail: 0.0,
            rail_d: FareSense::default(),
            len: LenOut::zero(),
            walk: 0.0,
            unr: 0.0,
            trips,
            round: 0,
            ms: 0.0,
            pairs: 0,
            sub: SubOut::empty(),
            fac,
            uniform: true,
            seen: vec![],
            xi,
            xfer,
            part: self.part,
        };
        self.staged_gpu = Some(StagedGpuPass {
            q,
            is_crowd: false,
            t0,
            period_idx,
            period,
            walk0,
            backup: None,
            sk_r,
            sk_walk_u32,
            sk_walk_len,
            sk_walk_max,
        });
        1
    }

    pub(crate) fn prepare_crowd_gpu_inner(&mut self, q: usize, total: &[f32]) -> i32 {
        if self.staged_gpu.is_some() {
            self.abort_gpu_inner();
        }
        let t0 = super::clock::now_ms();
        let st_part = {
            let Some(Some(st)) = self.per.get(q) else { return 0 };
            if round_costs(&st.g, &st.net, &st.p, total).is_none() {
                return 0;
            }
            st.part
        };
        if self.sz.is_none() {
            return if self.crowd_with_external(q, total) { 2 } else { 0 };
        }
        // Ensure stages exist for this period & `st.part` before mutating `self.per[q]`
        let saved_part = self.part;
        self.part = st_part;
        let (period_idx, max_acc_span, s, trivial) = match self.ensure_gpu_stages(q) {
            Some((idx, net_ref, per_ref)) => (
                idx,
                net_ref.max_acc_span,
                net_ref.s,
                per_ref.pairs < 32 || per_ref.n_visits < 64,
            ),
            None => {
                self.part = saved_part;
                return 0;
            }
        };
        self.part = saved_part;
        if trivial {
            return if self.crowd_with_external(q, total) { 2 } else { 0 };
        }

        let Some(mut st) = self.per.get_mut(q).and_then(|p| p.take()) else { return 0 };
        let Some((fac, left)) = round_costs(&st.g, &st.net, &st.p, total) else {
            self.per[q] = Some(st);
            return 0;
        };
        let seen_len = st.seen.len();
        if settled(&mut st.seen, &total[..st.g.rn_line.len()])
            || (st.uniform && fac == st.fac && left == st.g.rn_left)
        {
            st.round += 1;
            st.ms = super::clock::now_ms() - t0;
            self.per[q] = Some(st);
            return 2;
        }
        let backup = CrowdBackup {
            fac: st.fac.clone(),
            left: st.g.rn_left.clone(),
            uniform: st.uniform,
            seen: st.seen.clone(),
            ms: st.ms,
        };

        set_crowding(&mut st.g, &fac);
        st.g.rn_left = left;
        st.uniform = false;
        st.fac = fac;

        let (sk_r, sk_walk_u32, sk_walk_len, sk_walk_max) = with_skims(&st.g, &st.p, &self.want, |sk| {
            (
                sk.r.clone(),
                pack_sk_walk_u16(&sk.walk),
                sk.walk.len(),
                sk.walk_max.iter().map(|&v| v as f32).collect::<Vec<f32>>(),
            )
        });
        let rule = st.p.walk_rule > 0.0 && sk_walk_len == s * s && sk_walk_max.len() == s;
        if rule && max_acc_span > 64 {
            set_crowding(&mut st.g, &backup.fac);
            st.g.rn_left = backup.left;
            st.uniform = backup.uniform;
            st.fac = backup.fac;
            st.seen = backup.seen;
            st.seen.truncate(seen_len);
            st.ms = backup.ms;
            self.per[q] = Some(st);
            return if self.crowd_with_external(q, total) { 2 } else { 0 };
        }

        self.staged_gpu = Some(StagedGpuPass {
            q,
            is_crowd: true,
            t0,
            period_idx,
            period: st,
            walk0: 0.0,
            backup: Some(backup),
            sk_r,
            sk_walk_u32,
            sk_walk_len,
            sk_walk_max,
        });
        1
    }

    pub(crate) fn abort_gpu_inner(&mut self) {
        if let Some(mut staged) = self.staged_gpu.take() {
            if let Some(backup) = staged.backup {
                set_crowding(&mut staged.period.g, &backup.fac);
                staged.period.g.rn_left = backup.left;
                staged.period.uniform = backup.uniform;
                staged.period.fac = backup.fac;
                staged.period.seen = backup.seen;
                staged.period.ms = backup.ms;
                self.per[staged.q] = Some(staged.period);
            }
        }
    }

    /// Commit GPU `mode_choice` results into symmetric OD -> graph assignment -> MSA/Period.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn commit_gpu_inner(
        &mut self,
        od_in: &[f32],
        acc_home_in: &[f32],
        acc_q_flat: &[f32],
        rail_by_sub_u: &[f32],
        walk_lost_by_sub_u: &[f32],
        unr_lost_by_sub_u: &[f32],
        u_sense: &[f32],
        lost_q: &[f32],
        unr_q: &[f32],
        rail_band_in: &[f32],
        len_flat: &[f32],
    ) -> bool {
        let staged = match self.staged_gpu.take() {
            Some(s) => s,
            None => return false,
        };
        let (Some(net_stage), Some(per_stage), Some(sz)) = (
            self.gpu_net.as_ref(),
            self.gpu_per.get(staged.period_idx),
            self.sz.as_ref(),
        ) else {
            self.staged_gpu = Some(staged);
            self.abort_gpu_inner();
            return false;
        };

        let s = net_stage.s;
        let nu = net_stage.nu;
        let nq = net_stage.nq;
        let na_entries = net_stage.na_entries;
        let parted = net_stage.parted;
        let rule = staged.period.p.walk_rule > 0.0
            && staged.sk_walk_len == s * s
            && staged.sk_walk_max.len() == s;

        if od_in.len() != s * s
            || acc_home_in.len() != na_entries
            || acc_q_flat.len() != nq * KE
            || rail_by_sub_u.len() != nu
            || walk_lost_by_sub_u.len() != nu
            || u_sense.len() != nu * 3
            || lost_q.len() != nq
            || rail_band_in.len() != N_BANDS
            || len_flat.len() != 6 * N_LEN
        {
            self.staged_gpu = Some(staged);
            self.abort_gpu_inner();
            return false;
        }

        let mut od = od_in.to_vec();
        let mut acc_home = acc_home_in.to_vec();
        let mut acc_work = vec![0.0f32; na_entries];
        for q_i in 0..nq {
            let wn = net_stage.eg_n_flat[q_i] as usize;
            for kk in 0..wn {
                let k = net_stage.eg_k_flat[q_i * KE + kk] as usize;
                acc_work[k] = acc_q_flat[q_i * KE + kk];
            }
        }
        if rule {
            for x in od
                .iter_mut()
                .chain(acc_home.iter_mut())
                .chain(acc_work.iter_mut())
            {
                *x = x.max(0.0);
            }
        }

        let mut rail_by_sub = vec![0.0f32; sz.n];
        let mut walk_lost_by_sub = vec![0.0f32; sz.n];
        let mut unr_lost_by_sub = if parted { vec![0.0f32; sz.n] } else { vec![] };
        let mut rail = 0.0f64;
        let mut rail_walk = 0.0f64;
        let mut rail_unr = 0.0f64;
        let mut d_sum = 0.0f64;
        let mut dk_sum = 0.0f64;
        let mut rk_sum = 0.0f64;

        for u_ord in 0..nu {
            let u = net_stage.u_orig[u_ord] as usize;
            let rb = rail_by_sub_u[u_ord];
            let wl = walk_lost_by_sub_u[u_ord];
            rail_by_sub[u] = rb;
            walk_lost_by_sub[u] = wl;
            rail += rb as f64;
            rail_walk += wl as f64;
            d_sum += u_sense[u_ord * 3] as f64;
            dk_sum += u_sense[u_ord * 3 + 1] as f64;
            rk_sum += u_sense[u_ord * 3 + 2] as f64;
            if parted {
                let ul = unr_lost_by_sub_u[u_ord];
                unr_lost_by_sub[u] = ul;
                rail_unr += ul as f64;
            }
        }

        let mut rail_to_sub = vec![0.0f32; sz.n];
        for v in 0..sz.n {
            rail_to_sub[v] = acc_work[sz.acc_off[v] as usize..sz.acc_off[v + 1] as usize]
                .iter()
                .sum();
        }
        let mut walk_lost_to_sub = vec![0.0f32; sz.n];
        let mut unr_lost_to_sub = if parted { vec![0.0f32; sz.n] } else { vec![] };
        for q_i in 0..nq {
            let v = net_stage.dd_sub[q_i] as usize;
            walk_lost_to_sub[v] = lost_q[q_i];
            if parted {
                unr_lost_to_sub[v] = unr_q[q_i];
            }
        }

        let mut rail_band = [0.0f64; N_BANDS];
        for b in 0..N_BANDS {
            rail_band[b] = rail_band_in[b] as f64;
        }
        self.last_gpu_rail_band.set(rail_band);
        let _ = (per_stage.trips_access, rail_unr);

        let sense = FareSense {
            d: d_sum / FARE_D_MIN as f64,
            d_km: dk_sum / FARE_D_MIN as f64,
            km: rk_sum,
        };
        let mut len = LenOut::zero();
        for k in 0..(2 * N_LEN) {
            len.rail[k] = len_flat[k] as f64;
            len.walk[k] = len_flat[2 * N_LEN + k] as f64;
            len.unr[k] = len_flat[4 * N_LEN + k] as f64;
        }

        // Now run exact CPU `od_sym` + `assign_with_transfers` (hitting warm `SKIM_CACHE`)
        let q = staged.q;
        let (fw, fh) = od_sym_factors(q);
        let t_sym0 = super::clock::now_ms();
        let od_sub = self.compact_od(&od, s);
        for a in 0..s {
            for t in a + 1..s {
                let (x, y) = (od[a * s + t], od[t * s + a]);
                od[a * s + t] = fw * x + fh * y;
                od[t * s + a] = fw * y + fh * x;
            }
        }
        let t_as0 = super::clock::now_ms();
        let mut xfer = vec![0.0; staged.period.xi.len];
        let l = with_skims(&staged.period.g, &staged.period.p, &self.want, |sk| {
            assign_with_transfers(&staged.period.g, sk, &od, &staged.period.xi, &mut xfer)
        });
        let t_end = super::clock::now_ms();
        self.last_pass_ms.set([0.0, 0.0, 0.0, t_as0 - t_sym0, t_end - t_as0]);

        let sub = SubOut::from_parts(
            rail_by_sub,
            rail_to_sub,
            walk_lost_by_sub,
            walk_lost_to_sub,
            unr_lost_by_sub,
            unr_lost_to_sub,
            acc_home,
            acc_work,
            od_sub,
        );

        let now = t_end;
        let mut st = staged.period;
        if staged.is_crowd {
            st.round += 1;
            let r = st.round;
            msa(&mut st.xfer, &xfer, r);
            msa(&mut st.avg.seg, &l.seg, r);
            msa(&mut st.avg.board, &l.board, r);
            msa(&mut st.avg.alight, &l.alight, r);
            msa(&mut st.avg.rn_board, &l.rn_board, r);
            msa(&mut st.avg.rn_alight, &l.rn_alight, r);
            msa(&mut st.avg.entry, &l.entry, r);
            msa(&mut st.avg.exit, &l.exit, r);
            msa(&mut st.avg.rn_fare_x, &l.rn_fare_x, r);
            st.sub.msa(&sub, r);
            let w = 1.0 / (r as f64 + 1.0);
            st.rail += (rail - st.rail) * w;
            st.rail_d.toward(sense, w);
            st.len.toward(&len, w);
            let walk_new = self.walk_all * st.p.period_share as f64 - rail_walk;
            st.walk += (walk_new - st.walk) * w;
            if !self.unr_home.is_empty() {
                st.unr = self.unr_all * st.p.period_share as f64 - st.sub.unr_rail();
            }
            st.ms = now - staged.t0;
            self.per[q] = Some(st);
        } else {
            let walk0 = if st.part.0 == 0 { staged.walk0 } else { 0.0 };
            let unr0 = if st.part.0 == 0 {
                self.unr_all * st.p.period_share as f64
            } else {
                0.0
            };
            st.avg = l;
            st.rail = rail;
            st.rail_d = sense;
            st.len = len;
            st.walk = walk0 - rail_walk;
            st.unr = unr0 - sub.unr_rail();
            st.ms = now - staged.t0;
            st.pairs = per_stage.pairs;
            st.sub = sub;
            st.xfer = xfer;
            self.per[q] = Some(st);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demand::kernel::{
        build_access, build_graph, build_subzones, build_zones, gravity, skims, NoRail,
    };
    use crate::demand::synth::synth_city;

    #[test]
    fn gpu_stage_period_matches_mode_choice_counts() {
        let city = synth_city(1);
        let p = Params::default();
        let z = build_zones(&city, p.zone_m);
        let gr = gravity(&z, &p, 40, 1e-3);
        let nr = NoRail::new(&z, &p);
        let net = crate::demand::network::synth_network();
        let acc = build_access(&city, &net, &p);
        let sz = build_subzones(&city, &z, &acc, &net, &p);
        let g = build_graph(&net, &p);
        let sk = skims(&g, &p);
        let er = crate::demand::kernel::station_exp(&sk, &p);
        let mo = crate::demand::kernel::mode_choice_part(
            &z, &gr, &sz, &er, &sk.walk, &sk.walk_max, g.n_st, &p, &nr, (0, 1),
        );

        let net_st = GpuNetworkStage::build(1, &z, &gr, &sz, g.n_st, &p);
        let key = PeriodStageKey {
            q: 0,
            part: (0, 1),
            period_share_bits: p.period_share.to_bits(),
            commute_chunk_bits: p.commute_chunk.to_bits(),
            chunk_salt: p.chunk_salt,
            prune_trips_bits: p.prune_trips.to_bits(),
            net_epoch: 1,
        };
        let per_st = GpuPeriodStage::build(key, &net_st, &z, &gr, &sz, &nr, &p);
        assert_eq!(per_st.zone_pairs, mo.zone_pairs);
        assert_eq!(per_st.pairs, mo.pairs);
        assert!((per_st.trips_access - mo.trips_access).abs() <= 1e-3 * mo.trips_access.max(1.0));
        assert!((per_st.pruned_trips - mo.pruned_trips).abs() <= 1e-3 * mo.pruned_trips.max(1.0));
    }

    #[test]
    fn gpu_stage_records_max_acc_span_and_empty_s() {
        let city = synth_city(1);
        let mut p = Params::default();
        p.ride_lines = 70;
        p.walk_rule = 0.0;
        let z = build_zones(&city, p.zone_m);
        let gr = gravity(&z, &p, 40, 1e-3);

        let empty_net = crate::demand::network::Network {
            st_x: vec![],
            st_y: vec![],
            line_name: vec![],
            line_off: vec![0],
            line_stops: vec![],
            hop_min: vec![],
            hop_back: vec![],
            line_speed_kmh: vec![],
            line_headway_min: vec![],
            line_seats: vec![],
            line_rated: vec![],
            line_crush: vec![],
            st_group: vec![],
            stop_x: vec![],
            stop_y: vec![],
            line_wrap: vec![],
            span: vec![],
            st_ups: vec![],
            st_passage: vec![],
            catches: vec![],
            st_bus: vec![],
            st_tram: vec![],
        };
        let acc0 = build_access(&city, &empty_net, &p);
        let sz0 = build_subzones(&city, &z, &acc0, &empty_net, &p);
        let stage0 = GpuNetworkStage::build(1, &z, &gr, &sz0, 0, &p);
        assert_eq!(stage0.s, 0);
        assert_eq!(stage0.max_acc_span, 0);
        assert_eq!(stage0.meta[37], 0);

        // 70 distinct 2-station radial lines inside `ride_max_m` so a hub subzone has >64 access entries
        let n_lines = 70usize;
        let mut net70 = empty_net;
        for l in 0..n_lines {
            let ang = (l as f32) * (std::f32::consts::TAU / n_lines as f32);
            let a = (l * 2) as u32;
            let b = (l * 2 + 1) as u32;
            net70.st_x.push(ang.cos() * 400.0);
            net70.st_y.push(ang.sin() * 400.0);
            net70.st_x.push(ang.cos() * 1400.0);
            net70.st_y.push(ang.sin() * 1400.0);
            net70.line_name.push(format!("L{l}"));
            net70.line_stops.push(a);
            net70.line_stops.push(b);
            net70.line_off.push(net70.line_stops.len() as u32);
            net70.hop_min.extend_from_slice(&[2.0, 0.0]);
            net70.line_speed_kmh.push(40.0);
            net70.line_headway_min.push([5.0; PERIODS]);
            net70.line_seats.push(400.0);
            net70.line_crush.push(800.0);
        }
        let acc70 = build_access(&city, &net70, &p);
        let sz70 = build_subzones(&city, &z, &acc70, &net70, &p);
        let stage70 = GpuNetworkStage::build(2, &z, &gr, &sz70, net70.st_x.len(), &p);
        assert!(
            stage70.max_acc_span > 64,
            "expected max_acc_span > 64, got {}",
            stage70.max_acc_span
        );
        assert_eq!(stage70.meta[37] as usize, stage70.max_acc_span);
    }
}
