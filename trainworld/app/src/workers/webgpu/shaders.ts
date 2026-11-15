/**
 * WGSL compute kernels for WebGPU execution of `mode_choice_part`.
 *
 * All kernels share a single 256-byte `Uniforms` struct layout matching Rust's
 * `build_pass_uniforms_f32` plus JS batch fields (`[54..63]`), keep storage
 * bindings per stage <= 7 (under WebGPU's 8-buffer default limit), sanitize
 * `fast_ln`/`fast_exp` inputs for inactive `select` lanes, and preserve
 * deterministic canonical addition orders with zero `f32` atomics.
 */

export const GPU_U_BATCH = 64;
export const KE = 8;
export const N_BANDS = 13;
export const N_LEN = 58;
export const MAX_SLOW = 131072;
export const W_LEN = 512;
export const BATCH_U32_HDR = 324;

const UNIFORMS_WGSL = /* wgsl */ `
const KE: u32 = 8u;
const N_BANDS: u32 = 13u;
const N_LEN: u32 = 58u;
const MAX_SLOW: u32 = 131072u;
const W_LEN: u32 = 512u;
const BATCH_U32_HDR: u32 = 324u;
const WALK_REACH_KM: f32 = 10.0;
const WALK_SLIVER: f32 = 0.01;
const FARE_D_MIN: f32 = 1.0;
const SPREAD_HALF: i32 = 24;

struct Uniforms {
  s: u32,
  nb: u32,
  nq: u32,
  nu: u32,
  n_dz: u32,
  n_oz: u32,
  na_entries: u32,
  n_batches: u32,
  parted: u32,
  rule: u32,
  u_batch: u32,
  pair_walk_max: f32,
  fare_min: f32,
  car_own_km_min: f32,
  w_drive: f32,
  beta: f32,
  damp_min: f32,
  asc_rail: f32,
  asc_walk: f32,
  ratio_fade_hi: f32,
  ratio_kf: f32,
  road_local_km: f32,
  road_end_km: f32,
  road_open_min_per_km: f32,
  road_detour: f32,
  walk_rule: f32,
  lk_zone_km: f32,
  // net_f32 offsets (meta[10..18])
  off_bands: u32,
  off_dd_f32: u32,
  off_eg_e: u32,
  off_acc_e: u32,
  off_acc_d: u32,
  off_acc_walk: u32,
  off_u_c_u: u32,
  off_u_meta_f32: u32,
  off_lk_ent_f32: u32,
  // net_u32 offsets (meta[19..36])
  off_eg_st: u32,
  off_eg_k: u32,
  off_eg_n: u32,
  off_dd_dz_k: u32,
  off_dd_part: u32,
  off_acc_st: u32,
  off_acc_band: u32,
  off_st_eg_off: u32,
  off_st_eg_items: u32,
  off_u_meta_u32: u32,
  off_lk_off: u32,
  off_lk_ent_bin: u32,
  off_batch_meta_u32: u32,
  off_batch_active_a_off: u32,
  off_batch_active_a: u32,
  off_batch_a_acc_off: u32,
  off_batch_acc_items: u32,
  off_dz_q_range: u32,
  // Pass / batch dynamic slots
  n_visits: u32,
  off_ou_period: u32,
  off_oz_visit_off: u32,
  off_sk_walk_max: u32,
  batch_idx: u32,
  batch_v_start: u32,
  batch_v_end: u32,
  batch_active_a_lo: u32,
  batch_active_a_hi: u32,
  max_v_batch: u32,
};
`;

const MATH_WGSL = /* wgsl */ `
fn fast_exp(x: f32) -> f32 {
  let xc = select(-80.0, clamp(x, -80.0, 80.0), x == x);
  let t = xc * 1.4426950408889634;
  let i = floor(t);
  let f = t - i;
  let p = 1.0 + f * (0.6931472 + f * (0.2402265 + f * (0.05550411 + f * (0.009618129 + f * 0.0013333558))));
  let bits = bitcast<u32>(i32(i) + 127) << 23u;
  return bitcast<f32>(bits) * p;
}

fn fast_ln(x: f32) -> f32 {
  let xc = select(1e-30, max(x, 1e-30), x == x);
  let half_sqrt2: u32 = 0x3f3504f3u;
  let b = bitcast<u32>(xc) - half_sqrt2;
  let e = bitcast<i32>(b) >> 23u;
  let t = bitcast<f32>((b & 0x007fffffu) + half_sqrt2) - 1.0;
  let q = 1.00000468 + t * (-0.499903616 + t * (0.332567778 + t * (-0.253978318 + t * (0.220702996 + t * -0.143383126))));
  return f32(e) * 0.69314718056 + t * q;
}

// Compensated two-sum helper: returns (hi, lo)
fn two_sum_add(acc: vec2<f32>, v: f32) -> vec2<f32> {
  let s = acc.x + v;
  let bb = s - acc.x;
  let e = (acc.x - (s - bb)) + (v - bb);
  let lo = acc.y + e;
  let s2 = s + lo;
  return vec2<f32>(s2, lo - (s2 - s));
}

fn out_off_acc_home(u: Uniforms) -> u32 { return 0u; }
fn out_off_acc_q(u: Uniforms) -> u32 { return u.na_entries; }
fn out_off_rail_by_sub(u: Uniforms) -> u32 { return u.na_entries + u.nq * KE; }
fn out_off_walk_lost(u: Uniforms) -> u32 { return u.na_entries + u.nq * KE + u.nu; }
fn out_off_unr_lost(u: Uniforms) -> u32 { return u.na_entries + u.nq * KE + 2u * u.nu; }
fn out_off_u_sense(u: Uniforms) -> u32 { return u.na_entries + u.nq * KE + 3u * u.nu; }
fn out_off_lost_q(u: Uniforms) -> u32 { return u.na_entries + u.nq * KE + 6u * u.nu; }
fn out_off_unr_q(u: Uniforms) -> u32 { return u.na_entries + u.nq * KE + 6u * u.nu + u.nq; }
fn out_off_rail_band(u: Uniforms) -> u32 { return u.na_entries + u.nq * KE + 6u * u.nu + 2u * u.nq; }
fn out_off_len(u: Uniforms) -> u32 { return u.na_entries + u.nq * KE + 6u * u.nu + 2u * u.nq + N_BANDS; }
fn out_off_status(u: Uniforms) -> u32 { return out_off_len(u) + 6u * N_LEN; }

fn bf_off_mw() -> u32 { return 0u; }
fn bf_off_ab_row() -> u32 { return 64u; }
fn bf_off_ab_y(u: Uniforms) -> u32 { return 64u + u.u_batch * u.nb * 2u * u.s; }
fn bf_off_uq(u: Uniforms) -> u32 { return 64u + 2u * u.u_batch * u.nb * 2u * u.s; }
fn bf_off_aby_in(u: Uniforms) -> u32 { return bf_off_uq(u) + u.u_batch * u.nq * 8u; }
fn bf_off_len_part(u: Uniforms) -> u32 { return bf_off_aby_in(u) + u.u_batch * u.nq * 2u; }
fn bf_off_len_visit(u: Uniforms) -> u32 { return bf_off_len_part(u) + W_LEN * 12u * N_LEN; }

fn bu_off_cand() -> u32 { return BATCH_U32_HDR; }
fn bu_off_slow_recs(u: Uniforms) -> u32 { return BATCH_U32_HDR + 2u * u.u_batch * u.nq; }
fn bu_off_slow_sorted(u: Uniforms) -> u32 { return bu_off_slow_recs(u) + MAX_SLOW * 8u; }
fn bu_off_slow_compact(u: Uniforms) -> u32 { return bu_off_slow_sorted(u) + MAX_SLOW; }
fn bu_off_slow_bkt_cnt(u: Uniforms) -> u32 { return bu_off_slow_compact(u) + MAX_SLOW * 4u; }
fn bu_off_slow_bkt_bounds(u: Uniforms) -> u32 { return bu_off_slow_bkt_cnt(u) + 49152u; }
fn bu_off_slow_bkt_data(u: Uniforms) -> u32 { return bu_off_slow_bkt_bounds(u) + 256u; }
`;

export function computeBufferLayout(meta: Uint32Array, maxVBatch: number) {
  const s = meta[0];
  const nb = meta[1];
  const nq = meta[2];
  const nu = meta[3];
  const naEntries = meta[6];
  const uBatch = meta[9];

  const lenOutOff = naEntries + nq * KE + 6 * nu + 2 * nq + N_BANDS;
  const statusOff = lenOutOff + 6 * N_LEN;
  const outScalarsFloats = statusOff + 4;
  const outOffsets = {
    accHome: 0,
    accQ: naEntries,
    railBySub: naEntries + nq * KE,
    walkLostBySub: naEntries + nq * KE + nu,
    unrLostBySub: naEntries + nq * KE + 2 * nu,
    uSense: naEntries + nq * KE + 3 * nu,
    lostQ: naEntries + nq * KE + 6 * nu,
    unrQ: naEntries + nq * KE + 6 * nu + nq,
    railBand: naEntries + nq * KE + 6 * nu + 2 * nq,
    lenOut: lenOutOff,
    status: statusOff,
  };

  const strideAb = uBatch * nb * 2 * s;
  const uqFloats = uBatch * nq * 10;
  const lenPartOff = 64 + 2 * strideAb + uqFloats;
  const lenPartFloats = W_LEN * 12 * N_LEN;
  const batchF32Floats = Math.max(64, lenPartOff + lenPartFloats + maxVBatch * 3);
  const batchU32Words = BATCH_U32_HDR + 2 * uBatch * nq + MAX_SLOW * 19 + 49408;
  const erFloats = Math.max(1, nb * s * s);
  const odFloats = Math.max(1, s * s);

  return {
    outScalarsFloats,
    outOffsets,
    lenPartOff,
    lenPartFloats,
    batchF32Floats,
    batchU32Words,
    erFloats,
    odFloats,
  };
}

export const STATION_EXP_WGSL = /* wgsl */ `
${UNIFORMS_WGSL}
${MATH_WGSL}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> net_f32: array<f32>;
@group(0) @binding(2) var<storage, read> skim_f32: array<f32>;
@group(0) @binding(3) var<storage, read_write> er_buf: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let ss = u.s * u.s;
  let total = u.nb * ss;
  let idx = (wid.y * nwg.x + wid.x) * 256u + lid.x;
  if (idx >= total) {
    return;
  }
  let b = idx / ss;
  let rem = idx - b * ss;
  let a = rem / u.s;
  let t = rem - a * u.s;
  if (a == t) {
    er_buf[idx] = 0.0;
    return;
  }
  let r = skim_f32[rem];
  if (!(r < 1e30)) {
    er_buf[idx] = 0.0;
    return;
  }
  let th = net_f32[u.off_bands + b * 2u];
  let off = net_f32[u.off_bands + b * 2u + 1u];
  er_buf[idx] = fast_exp(-th * (r - off));
}
`;

export const ZERO_PASS_WGSL = /* wgsl */ `
${UNIFORMS_WGSL}
${MATH_WGSL}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> od_buf: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_scalars: array<f32>;
@group(0) @binding(3) var<storage, read_write> batch_f32: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let idx = (wid.y * nwg.x + wid.x) * 256u + lid.x;
  let od_len = u.s * u.s;
  let out_len = out_off_status(u) + 4u;
  let len_p_len = W_LEN * 12u * N_LEN;
  if (idx < od_len) {
    od_buf[idx] = 0.0;
  }
  if (idx < out_len) {
    out_scalars[idx] = 0.0;
  }
  if (idx < len_p_len) {
    batch_f32[bf_off_len_part(u) + idx] = 0.0;
  }
}
`;

export const UQ_ZERO_WGSL = /* wgsl */ `
${UNIFORMS_WGSL}
${MATH_WGSL}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> batch_f32: array<f32>;
@group(0) @binding(2) var<storage, read_write> batch_u32: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let idx = (wid.y * nwg.x + wid.x) * 256u + lid.x;
  let total_uq = u.u_batch * u.nq * 10u;
  if (idx < total_uq) {
    batch_f32[bf_off_uq(u) + idx] = 0.0;
  }
  if (idx < BATCH_U32_HDR) {
    batch_u32[idx] = 0u;
  }
  let v_cnt = u.batch_v_end - u.batch_v_start;
  if (idx < v_cnt * 3u) {
    batch_f32[bf_off_len_visit(u) + idx] = 0.0;
  }
}
`;

export const U_SETUP_WGSL = /* wgsl */ `
${UNIFORMS_WGSL}
${MATH_WGSL}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> net_f32: array<f32>;
@group(0) @binding(2) var<storage, read> net_u32: array<u32>;
@group(0) @binding(3) var<storage, read> per_f32: array<f32>;
@group(0) @binding(4) var<storage, read> per_u32: array<u32>;
@group(0) @binding(5) var<storage, read> skim_f32: array<f32>;
@group(0) @binding(6) var<storage, read_write> batch_f32: array<f32>;
@group(0) @binding(7) var<storage, read_write> batch_u32: array<u32>;

var<workgroup> sh_mw: array<f32, 64>;
var<workgroup> sh_nv: array<u32, 64>;
var<workgroup> sh_used: array<u32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let u_local = wid.x;
  let tid = lid.x;
  let bm = u.off_batch_meta_u32 + u.batch_idx * 4u;
  let u_start = net_u32[bm + 0u];
  let u_end = net_u32[bm + 1u];
  let u_ord = u_start + u_local;
  if (u_ord >= u_end) {
    return;
  }

  let mu_base = u.off_u_meta_u32 + u_ord * 8u;
  let mf_base = u.off_u_meta_f32 + u_ord * 8u;
  let oi = net_u32[mu_base + 1u];
  let walk_start = net_u32[mu_base + 2u];
  let walk_end = net_u32[mu_base + 3u];
  let ride_end = net_u32[mu_base + 4u];
  let rides = ride_end > walk_end;

  let m_u = net_f32[mf_base + 5u];
  let ride_b = net_f32[mf_base + 7u];

  var mw: f32 = 0.0;
  if (u.rule != 0u) {
    for (var k = walk_start + tid; k < ride_end; k = k + 64u) {
      let a = net_u32[u.off_acc_st + k];
      let v = net_f32[u.off_acc_walk + k] + skim_f32[u.off_sk_walk_max + a];
      mw = max(mw, v);
    }
    sh_mw[tid] = mw;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
      if (tid < stride) {
        sh_mw[tid] = max(sh_mw[tid], sh_mw[tid + stride]);
      }
      workgroupBarrier();
    }
    if (tid == 0u) {
      batch_f32[bf_off_mw() + u_local] = sh_mw[0];
    }
  } else if (tid == 0u) {
    batch_f32[bf_off_mw() + u_local] = 0.0;
  }

  let v_lo = per_u32[u.off_oz_visit_off + oi];
  let v_hi = per_u32[u.off_oz_visit_off + oi + 1u];
  let wmax_u = u.pair_walk_max - m_u;
  let wmax_r = u.pair_walk_max - ride_b;
  let lim_u = select(wmax_u, wmax_r, rides);

  var used: u32 = 0u;
  var cand_cnt: u32 = 0u;
  let cand_base = bu_off_cand() + u_local * u.nq * 2u;

  for (var v_base = v_lo; v_base < v_hi; v_base = v_base + 64u) {
    let v = v_base + tid;
    var nv_v: u32 = 0u;
    var used_v: u32 = 0u;
    var q0_v: u32 = 0u;
    var zv_v: u32 = 0u;

    if (v < v_hi) {
      let vu_base = v * 4u;
      let zk = per_u32[vu_base + 0u];
      let meta_bits = per_u32[vu_base + 1u];
      let zb = meta_bits & 0xffu;
      let long_f = ((meta_bits >> 8u) & 1u) != 0u;
      let by_ride = rides && long_f;
      let wmax = select(wmax_u, wmax_r, by_ride);

      let q0 = net_u32[u.off_dz_q_range + 2u * zk];
      let q1 = net_u32[u.off_dz_q_range + 2u * zk + 1u];
      if (q1 > q0) {
        let zmin = net_f32[u.off_dd_f32 + q0 * 12u + 6u];
        let step_floor = f32(u32(max(zmin, 0.0) * 0.1)) * 10.0;
        if (step_floor <= lim_u && zmin <= wmax) {
          let zmax = net_f32[u.off_dd_f32 + (q1 - 1u) * 12u + 6u];
          var nv: u32 = q1 - q0;
          if (zmax > wmax) {
            var l: u32 = 0u;
            var r: u32 = q1 - q0;
            loop {
              if (l >= r) {
                break;
              }
              let mid = l + (r - l) / 2u;
              if (net_f32[u.off_dd_f32 + (q0 + mid) * 12u + 6u] <= wmax) {
                l = mid + 1u;
              } else {
                r = mid;
              }
            }
            nv = l;
          }
          if (nv > 0u) {
            nv_v = nv;
            used_v = 1u << zb;
            q0_v = q0;
            zv_v = v | select(0u, 1u << 31u, by_ride);
          }
        }
      }
    }

    sh_nv[tid] = nv_v;
    sh_used[tid] = used_v;
    workgroupBarrier();

    for (var off = 1u; off < 64u; off = off << 1u) {
      var add_nv: u32 = 0u;
      var or_u: u32 = 0u;
      if (tid >= off) {
        add_nv = sh_nv[tid - off];
        or_u = sh_used[tid - off];
      }
      workgroupBarrier();
      if (tid >= off) {
        sh_nv[tid] = sh_nv[tid] + add_nv;
        sh_used[tid] = sh_used[tid] | or_u;
      }
      workgroupBarrier();
    }

    if (nv_v > 0u) {
      let local_off = sh_nv[tid] - nv_v;
      let dst = cand_base + (cand_cnt + local_off) * 2u;
      for (var qi = 0u; qi < nv_v; qi = qi + 1u) {
        let c_pos = dst + qi * 2u;
        batch_u32[c_pos + 0u] = q0_v + qi;
        batch_u32[c_pos + 1u] = zv_v;
      }
    }

    used = used | sh_used[63u];
    cand_cnt = cand_cnt + sh_nv[63u];
    workgroupBarrier();
  }

  if (tid == 0u) {
    batch_u32[u_local] = used;
    batch_u32[64u + u_local] = cand_cnt;
  }
}
`;

export const BUILD_AB_ROW_WGSL = /* wgsl */ `
${UNIFORMS_WGSL}
${MATH_WGSL}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> net_f32: array<f32>;
@group(0) @binding(2) var<storage, read> net_u32: array<u32>;
@group(0) @binding(3) var<storage, read> er_buf: array<f32>;
@group(0) @binding(4) var<storage, read_write> batch_f32: array<f32>;
@group(0) @binding(5) var<storage, read> batch_u32: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  let b = gid.y;
  let u_local = gid.z;
  if (t >= u.s || b >= u.nb) {
    return;
  }
  let bm = u.off_batch_meta_u32 + u.batch_idx * 4u;
  let u_start = net_u32[bm + 0u];
  let u_end = net_u32[bm + 1u];
  let u_ord = u_start + u_local;
  if (u_ord >= u_end) {
    return;
  }

  let row_base = bf_off_ab_row() + ((u_local * u.nb + b) * 2u) * u.s + t;
  let used = batch_u32[u_local];
  if (((used >> b) & 1u) == 0u) {
    batch_f32[row_base] = 0.0;
    batch_f32[row_base + u.s] = 0.0;
    return;
  }

  let mu_base = u.off_u_meta_u32 + u_ord * 8u;
  let walk_start = net_u32[mu_base + 2u];
  let walk_end = net_u32[mu_base + 3u];
  let ride_end = net_u32[mu_base + 4u];
  let rides = ride_end > walk_end;

  let ss = u.s * u.s;
  let er_b_off = b * ss + t;
  var a_val: f32 = 0.0;
  for (var k = walk_start; k < walk_end; k = k + 1u) {
    let a = net_u32[u.off_acc_st + k];
    let w = net_f32[u.off_acc_e + b * u.na_entries + k];
    a_val = a_val + w * er_buf[er_b_off + a * u.s];
  }
  batch_f32[row_base] = a_val;

  if (rides) {
    let c_ub = net_f32[u.off_u_c_u + u_ord * u.nb + b];
    var b_val = c_ub * a_val;
    for (var k = walk_end; k < ride_end; k = k + 1u) {
      let a = net_u32[u.off_acc_st + k];
      let w = net_f32[u.off_acc_e + b * u.na_entries + k];
      b_val = b_val + w * er_buf[er_b_off + a * u.s];
    }
    batch_f32[row_base + u.s] = b_val;
  } else {
    batch_f32[row_base + u.s] = 0.0;
  }
}
`;

export const MODE_PAIRS_WGSL = /* wgsl */ `
${UNIFORMS_WGSL}
${MATH_WGSL}

struct ModeBatchU32 {
  hdr: array<atomic<u32>, 324>,
  data: array<u32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> net_f32: array<f32>;
@group(0) @binding(2) var<storage, read> net_u32: array<u32>;
@group(0) @binding(3) var<storage, read> per_f32: array<f32>;
@group(0) @binding(4) var<storage, read> er_buf: array<f32>;
@group(0) @binding(5) var<storage, read> skim_u32: array<u32>;
@group(0) @binding(6) var<storage, read_write> batch_f32: array<f32>;
@group(0) @binding(7) var<storage, read_write> batch_u32: ModeBatchU32;

fn ratio_share(rail_c: f32, car: f32, hi: f32, kf: f32) -> f32 {
  let qr = rail_c / car;
  let s = 1.0 / (1.0 + qr * qr * qr);
  let fr = clamp((hi - qr) * kf, 0.0, 1.0);
  let fc = clamp((hi - 1.0 / qr) * kf, 0.0, 1.0);
  return 1.0 - (1.0 - s * fr) * fc;
}

fn car_choice(
  rail: f32,
  bt: f32,
  road: f32,
  paid: f32,
  free: f32,
  uw: f32,
  asc_rail: f32,
  hi: f32,
  kf: f32,
  paying: bool,
) -> f32 {
  let util = bt * (road - rail) + asc_rail;
  let rail_c = max(rail, 0.01);
  let e = fast_exp(uw - util);
  let w0 = e / (1.0 + fast_exp(-util) + e);
  let s0 = (1.0 - w0) * ratio_share(rail_c, road, hi, kf);
  if (!paying) {
    return s0;
  }
  let w1 = e / (1.0 + fast_exp(-(util + bt * paid)) + e);
  let s1 = (1.0 - w1) * ratio_share(rail_c, road + paid, hi, kf);
  return s0 + (1.0 - free) * (s1 - s0);
}

fn bound_share(rail: f32, road: f32, paid: f32, free: f32, hi: f32, kf: f32, paying: bool) -> f32 {
  let rail_c = max(rail, 0.01);
  let f0 = clamp((hi - rail_c / road) * kf, 0.0, 1.0);
  if (!paying) {
    return f0;
  }
  let f1 = clamp((hi - rail_c / (road + paid)) * kf, 0.0, 1.0);
  return f0 + (1.0 - free) * (f1 - f0);
}

fn read_sk_walk(a: u32, t: u32, s: u32) -> f32 {
  let idx = a * s + t;
  let word = skim_u32[idx >> 1u];
  let val = select(word & 0xffffu, word >> 16u, (idx & 1u) != 0u);
  return f32(val);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pair_idx = gid.x;
  let u_local = gid.y;
  let bm = u.off_batch_meta_u32 + u.batch_idx * 4u;
  let u_start = net_u32[bm + 0u];
  let u_end = net_u32[bm + 1u];
  let u_ord = u_start + u_local;
  if (u_ord >= u_end) {
    return;
  }
  let cand_cnt = atomicLoad(&batch_u32.hdr[64u + u_local]);
  if (pair_idx >= cand_cnt) {
    return;
  }

  let c_pos = (u_local * u.nq + pair_idx) * 2u;
  let q = batch_u32.data[c_pos + 0u];
  let zv = batch_u32.data[c_pos + 1u];
  let v = zv & 0x7fffffffu;
  let by_ride = (zv >> 31u) != 0u;

  let vf_base = v * 8u;
  let f_m = per_f32[vf_base + 0u];
  let cw = per_f32[vf_base + 1u];
  let v_df = per_f32[vf_base + 2u];
  let w0 = per_f32[vf_base + 3u];
  let wf = per_f32[vf_base + 4u];
  let b = bitcast<u32>(per_f32[vf_base + 6u]);

  let ou_u = per_f32[u.off_ou_period + u_ord];
  let tu = ou_u * f_m;

  let mu_base = u.off_u_meta_u32 + u_ord * 8u;
  let mf_base = u.off_u_meta_f32 + u_ord * 8u;
  let walk_start = net_u32[mu_base + 2u];
  let walk_end = net_u32[mu_base + 3u];
  let ride_end = net_u32[mu_base + 4u];
  let pu = net_u32[mu_base + 5u];

  let ux = net_f32[mf_base + 0u];
  let uy = net_f32[mf_base + 1u];
  let lu0 = net_f32[mf_base + 2u];
  let lu1 = net_f32[mf_base + 3u];
  let hp = net_f32[mf_base + 4u];
  let m_u = net_f32[mf_base + 5u];
  let m_r = net_f32[mf_base + 6u];
  let mo = select(m_u, m_r, by_ride);

  let dd_base = u.off_dd_f32 + q * 12u;
  let vx = net_f32[dd_base + 0u];
  let vy = net_f32[dd_base + 1u];
  let l0 = net_f32[dd_base + 2u];
  let l1 = net_f32[dd_base + 3u];
  let park = net_f32[dd_base + 4u];
  let dv = net_f32[dd_base + 5u];
  let m8 = net_f32[dd_base + 6u];
  let paid = net_f32[dd_base + 7u];
  let free = net_f32[dd_base + 8u];
  let eg_far = net_f32[dd_base + 9u];

  let cross = (u.parted != 0u) && (net_u32[u.off_dd_part + q] != pu);
  let rr = select(0u, 1u, by_ride);
  let ab_base = bf_off_ab_row() + ((u_local * u.nb + b) * 2u + rr) * u.s;

  let ne = net_u32[u.off_eg_n + q];
  var rv: array<f32, 8>;
  var ev: array<f32, 8>;
  var ssum: f32 = 0.0;
  for (var kk = 0u; kk < ne; kk = kk + 1u) {
    let t = net_u32[u.off_eg_st + q * KE + kk];
    let r = batch_f32[ab_base + t];
    let e = net_f32[u.off_eg_e + (b * u.nq + q) * KE + kk];
    rv[kk] = r;
    ev[kk] = e;
    ssum = ssum + r * e;
  }

  let dx = ux - vx;
  let dy = uy - vy;
  let d = sqrt(dx * dx + dy * dy);
  let mw = batch_f32[bf_off_mw() + u_local];
  let d_lim = min(mw + eg_far, WALK_REACH_KM * 1000.0 + 1.0);
  let direct = d * wf;
  let near = (u.rule != 0u) && (!cross) && (d < d_lim) && (u.walk_rule > 0.0) && (direct <= WALK_REACH_KM * 1000.0);

  var by_j_packed: array<u32, 16>;
  for (var i = 0u; i < 16u; i = i + 1u) {
    by_j_packed[i] = 0u;
  }
  var slow_here: bool = false;
  var most_here: bool = false;
  var nk: u32 = 0u;

  if (near && ssum >= 1e-30) {
    nk = select(walk_end - walk_start, ride_end - walk_start, by_ride);
    var slow_k_any: u32 = 0u;
    var bound_sum: f32 = 0.0;
    for (var kk = 0u; kk < ne; kk = kk + 1u) {
      let t = net_u32[u.off_eg_st + q * KE + kk];
      let wt = net_f32[u.off_acc_d + net_u32[u.off_eg_k + q * KE + kk]];
      var any_kk: bool = false;
      for (var j = 0u; j < nk; j = j + 1u) {
        let k = walk_start + j;
        let a = net_u32[u.off_acc_st + k];
        let c = net_f32[u.off_acc_walk + k] + read_sk_walk(a, t, u.s);
        if (c + wt > direct) {
          if (j < 64u) {
            let w_idx = j >> 2u;
            let shift = (j & 3u) * 8u + kk;
            by_j_packed[w_idx] = by_j_packed[w_idx] | (1u << shift);
          }
          any_kk = true;
        }
      }
      if (any_kk) {
        slow_k_any = slow_k_any | (1u << kk);
        bound_sum = bound_sum + rv[kk] * ev[kk];
      }
    }

    if (slow_k_any != 0u && bound_sum > 0.9999 * WALK_SLIVER * ssum) {
      let c_b = select(1.0, net_f32[u.off_u_c_u + u_ord * u.nb + b], by_ride);
      let er_b_base = b * u.s * u.s;
      var bad: f32 = 0.0;
      for (var j = 0u; j < nk; j = j + 1u) {
        let km = (by_j_packed[j >> 2u] >> ((j & 3u) * 8u)) & 0xffu;
        if (km == 0u) {
          continue;
        }
        let k = walk_start + j;
        let a = net_u32[u.off_acc_st + k];
        let w = net_f32[u.off_acc_e + b * u.na_entries + k] * select(1.0, c_b, k < walk_end);
        for (var kk = 0u; kk < ne; kk = kk + 1u) {
          if (((km >> kk) & 1u) != 0u) {
            let t = net_u32[u.off_eg_st + q * KE + kk];
            let cw_pair = w * er_buf[er_b_base + a * u.s + t] * net_f32[u.off_eg_e + (b * u.nq + q) * KE + kk];
            if (cw_pair > 0.0) {
              bad = bad + cw_pair;
            }
          }
        }
      }

      if (bad > WALK_SLIVER * ssum) {
        slow_here = true;
        most_here = bad > 0.5 * ssum;
        if (most_here) {
          var kept: f32 = 0.0;
          let all_k = (1u << ne) - 1u;
          for (var j = 0u; j < nk; j = j + 1u) {
            let byj = (by_j_packed[j >> 2u] >> ((j & 3u) * 8u)) & 0xffu;
            let km = (~byj) & all_k;
            if (km == 0u) {
              continue;
            }
            let k = walk_start + j;
            let a = net_u32[u.off_acc_st + k];
            let w = net_f32[u.off_acc_e + b * u.na_entries + k] * select(1.0, c_b, k < walk_end);
            for (var kk = 0u; kk < ne; kk = kk + 1u) {
              if (((km >> kk) & 1u) != 0u) {
                let t = net_u32[u.off_eg_st + q * KE + kk];
                let cw_pair = w * er_buf[er_b_base + a * u.s + t] * net_f32[u.off_eg_e + (b * u.nq + q) * KE + kk];
                if (cw_pair > 0.0) {
                  kept = kept + cw_pair;
                }
              }
            }
          }
          ssum = ssum * (kept / ssum);
        } else {
          ssum = ssum * (1.0 - bad / ssum);
        }
      }
    }
  }

  if (!(ssum >= 1e-30)) {
    return;
  }

  let df = select(v_df, u.road_detour, cross);
  let rd = d * 0.001 * df;
  let e_seg = min(0.5 * rd, u.road_local_km);
  let main_seg = rd - 2.0 * e_seg;
  let k_seg = min(0.5 * main_seg, u.road_end_km);
  let drive = e_seg * (lu0 + l0) + k_seg * (lu1 + l1) + (main_seg - 2.0 * k_seg) * u.road_open_min_per_km;
  let road = drive * u.w_drive + rd * u.car_own_km_min + park + hp;

  let th = net_f32[u.off_bands + b * 2u];
  let off = net_f32[u.off_bands + b * 2u + 1u];
  let rail = mo + m8 + u.fare_min - fast_ln(ssum) / th + off;

  let bt = u.beta * sqrt(min(u.damp_min / road, 1.0));
  let uw = bt * (road - cw) + u.asc_walk;
  let paying = paid >= 1e-30;

  let s0 = car_choice(rail, bt, road, paid, free, uw, u.asc_rail, u.ratio_fade_hi, u.ratio_kf, paying);
  let sh = select(s0, bound_share(rail, road, paid, free, u.ratio_fade_hi, u.ratio_kf, paying), cross);
  let x = tu * dv * sh;

  let rail_d = rail + FARE_D_MIN;
  let sd = car_choice(rail_d, bt, road, paid, free, uw, u.asc_rail, u.ratio_fade_hi, u.ratio_kf, paying);
  let shd = select(sd, bound_share(rail_d, road, paid, free, u.ratio_fade_hi, u.ratio_kf, paying), cross);
  let xd = tu * dv * shd - x;
  let dk = d * 0.001;

  let g = x / ssum;
  let wl = select(x * w0, 0.0, cross);
  let unr_x = select(0.0, x, cross);

  let g_eff = select(g, 0.0, most_here);
  let uq_base = bf_off_uq(u) + (u_local * u.nq + q) * 8u;
  batch_f32[uq_base + 0u] = g_eff;
  batch_f32[uq_base + 1u] = wl;
  batch_f32[uq_base + 2u] = unr_x;
  batch_f32[uq_base + 3u] = x;
  batch_f32[uq_base + 4u] = xd;
  batch_f32[uq_base + 5u] = xd * dk;
  batch_f32[uq_base + 6u] = x * dk;
  let meta_u = b | (rr << 4u) | (select(0u, 1u, most_here) << 8u);
  batch_f32[uq_base + 7u] = bitcast<f32>(meta_u);

  if (g_eff > 0.0) {
    let aby_in_base = bf_off_aby_in(u) + (q * u.u_batch + u_local) * 2u;
    batch_f32[aby_in_base + 0u] = g_eff;
    batch_f32[aby_in_base + 1u] = bitcast<f32>(meta_u);
  }

  if (slow_here) {
    let c_b = select(1.0, net_f32[u.off_u_c_u + u_ord * u.nb + b], by_ride);
    let er_b_base = b * u.s * u.s;
    let all_k = (1u << ne) - 1u;
    var emit_cnt: u32 = 0u;
    for (var j = 0u; j < nk; j = j + 1u) {
      let byj = (by_j_packed[j >> 2u] >> ((j & 3u) * 8u)) & 0xffu;
      let km = select(byj, (~byj) & all_k, most_here);
      if (km == 0u) {
        continue;
      }
      let k = walk_start + j;
      let a = net_u32[u.off_acc_st + k];
      let w = net_f32[u.off_acc_e + b * u.na_entries + k] * select(1.0, c_b, k < walk_end);
      for (var kk = 0u; kk < ne; kk = kk + 1u) {
        if (((km >> kk) & 1u) != 0u) {
          let t = net_u32[u.off_eg_st + q * KE + kk];
          let cw_pair = w * er_buf[er_b_base + a * u.s + t] * net_f32[u.off_eg_e + (b * u.nq + q) * KE + kk];
          if (cw_pair > 0.0) {
            emit_cnt = emit_cnt + 1u;
          }
        }
      }
    }

    if (emit_cnt > 0u) {
      let base_slot = atomicAdd(&batch_u32.hdr[128u], emit_cnt);
      if (base_slot + emit_cnt > MAX_SLOW) {
        atomicMax(&batch_u32.hdr[129u], 1u);
        return;
      }
      let sign = select(-1.0, 1.0, most_here);
      let recs_base = bu_off_slow_recs(u) - BATCH_U32_HDR;
      var seq: u32 = 0u;
      for (var j = 0u; j < nk; j = j + 1u) {
        let byj = (by_j_packed[j >> 2u] >> ((j & 3u) * 8u)) & 0xffu;
        let km = select(byj, (~byj) & all_k, most_here);
        if (km == 0u) {
          continue;
        }
        let k = walk_start + j;
        let a = net_u32[u.off_acc_st + k];
        let w = net_f32[u.off_acc_e + b * u.na_entries + k] * select(1.0, c_b, k < walk_end);
        for (var kk = 0u; kk < ne; kk = kk + 1u) {
          if (((km >> kk) & 1u) != 0u) {
            let t = net_u32[u.off_eg_st + q * KE + kk];
            let cw_pair = w * er_buf[er_b_base + a * u.s + t] * net_f32[u.off_eg_e + (b * u.nq + q) * KE + kk];
            if (cw_pair > 0.0) {
              let rec = recs_base + (base_slot + seq) * 8u;
              batch_u32.data[rec + 0u] = (u_local << 24u) | (pair_idx & 0x00ffffffu);
              batch_u32.data[rec + 1u] = seq;
              batch_u32.data[rec + 2u] = a;
              batch_u32.data[rec + 3u] = t;
              batch_u32.data[rec + 4u] = q;
              batch_u32.data[rec + 5u] = k;
              batch_u32.data[rec + 6u] = kk;
              batch_u32.data[rec + 7u] = bitcast<u32>(sign * g * cw_pair);
              seq = seq + 1u;
            }
          }
        }
      }
    }
  }
}
`;

export const BUILD_AB_Y_WGSL = /* wgsl */ `
${UNIFORMS_WGSL}
${MATH_WGSL}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> net_f32: array<f32>;
@group(0) @binding(2) var<storage, read> net_u32: array<u32>;
@group(0) @binding(3) var<storage, read_write> batch_f32: array<f32>;
@group(0) @binding(4) var<storage, read> batch_u32: array<u32>;

var<workgroup> sh_act: array<u32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let u_local = lid.x;
  let t = wid.y;
  let b_side = wid.z;
  if (t >= u.s || b_side >= 2u * u.nb) {
    return;
  }
  let b = b_side >> 1u;
  let rr = b_side & 1u;
  let expected = b | (rr << 4u);

  let bm = u.off_batch_meta_u32 + u.batch_idx * 4u;
  let u_start = net_u32[bm + 0u];
  let u_end = net_u32[bm + 1u];
  let batch_nu = u_end - u_start;

  var is_active: bool = false;
  if (u_local < batch_nu) {
    let used = batch_u32[u_local];
    if (((used >> b) & 1u) != 0u) {
      if (rr == 0u) {
        is_active = true;
      } else {
        let u_ord = u_start + u_local;
        let mu_base = u.off_u_meta_u32 + u_ord * 8u;
        is_active = net_u32[mu_base + 4u] > net_u32[mu_base + 3u];
      }
    }
  }

  sh_act[u_local] = select(0u, 1u, is_active);
  workgroupBarrier();
  if (u_local < 32u) { sh_act[u_local] = sh_act[u_local] | sh_act[u_local + 32u]; }
  workgroupBarrier();
  if (u_local < 16u) { sh_act[u_local] = sh_act[u_local] | sh_act[u_local + 16u]; }
  workgroupBarrier();
  if (u_local < 8u) { sh_act[u_local] = sh_act[u_local] | sh_act[u_local + 8u]; }
  workgroupBarrier();
  if (u_local < 4u) { sh_act[u_local] = sh_act[u_local] | sh_act[u_local + 4u]; }
  workgroupBarrier();
  if (u_local < 2u) { sh_act[u_local] = sh_act[u_local] | sh_act[u_local + 2u]; }
  workgroupBarrier();
  if (u_local == 0u) {
    sh_act[0] = sh_act[0] | sh_act[1];
  }
  let any_active = workgroupUniformLoad(&sh_act[0]);

  if (any_active == 0u) {
    if (u_local < batch_nu) {
      batch_f32[bf_off_ab_y(u) + ((u_local * u.nb + b) * 2u + rr) * u.s + t] = 0.0;
    }
    return;
  }

  let s_lo = net_u32[u.off_st_eg_off + t];
  let s_hi = net_u32[u.off_st_eg_off + t + 1u];
  var acc: f32 = 0.0;
  let aby_in_off = bf_off_aby_in(u);
  for (var idx = s_lo; idx < s_hi; idx = idx + 1u) {
    let q = net_u32[u.off_st_eg_items + idx * 2u];
    let kk = net_u32[u.off_st_eg_items + idx * 2u + 1u];
    let e = net_f32[u.off_eg_e + (b * u.nq + q) * KE + kk];
    if (is_active) {
      let p = aby_in_off + (q * u.u_batch + u_local) * 2u;
      let g = batch_f32[p + 0u];
      if (g > 0.0) {
        let meta_u = bitcast<u32>(batch_f32[p + 1u]);
        if ((meta_u & 0x1ffu) == expected) {
          acc = acc + g * e;
        }
      }
    }
  }
  if (u_local < batch_nu) {
    batch_f32[bf_off_ab_y(u) + ((u_local * u.nb + b) * 2u + rr) * u.s + t] = acc;
  }
}
`;

export const SPLIT_TO_OD_WGSL = /* wgsl */ `
${UNIFORMS_WGSL}
${MATH_WGSL}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> net_f32: array<f32>;
@group(0) @binding(2) var<storage, read> net_u32: array<u32>;
@group(0) @binding(3) var<storage, read> er_buf: array<f32>;
@group(0) @binding(4) var<storage, read> batch_f32: array<f32>;
@group(0) @binding(5) var<storage, read> batch_u32: array<u32>;
@group(0) @binding(6) var<storage, read_write> od_buf: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  let ai = u.batch_active_a_lo + gid.y;
  if (t >= u.s || ai >= u.batch_active_a_hi) {
    return;
  }
  let bm = u.off_batch_meta_u32 + u.batch_idx * 4u;
  let u_start = net_u32[bm + 0u];
  let a = net_u32[u.off_batch_active_a + ai];
  let item_lo = net_u32[u.off_batch_a_acc_off + ai];
  let item_hi = net_u32[u.off_batch_a_acc_off + ai + 1u];

  let ss = u.s * u.s;
  let aby_off = bf_off_ab_y(u);
  var od_val = od_buf[a * u.s + t];

  for (var idx = item_lo; idx < item_hi; idx = idx + 1u) {
    let item0 = net_u32[u.off_batch_acc_items + idx * 2u];
    let k = net_u32[u.off_batch_acc_items + idx * 2u + 1u];
    let u_local = item0 & 0x7fffffffu;
    let is_walk = (item0 >> 31u) != 0u;
    let u_ord = u_start + u_local;
    let used = batch_u32[u_local];
    let mu_base = u.off_u_meta_u32 + u_ord * 8u;
    let rides = net_u32[mu_base + 4u] > net_u32[mu_base + 3u];

    for (var b = 0u; b < u.nb; b = b + 1u) {
      if (((used >> b) & 1u) == 0u) {
        continue;
      }
      let er_v = er_buf[b * ss + a * u.s + t];
      let w = net_f32[u.off_acc_e + b * u.na_entries + k];
      let yb_base = aby_off + ((u_local * u.nb + b) * 2u) * u.s + t;
      var yv: f32;
      if (is_walk) {
        yv = batch_f32[yb_base];
        if (rides) {
          yv = yv + net_f32[u.off_u_c_u + u_ord * u.nb + b] * batch_f32[yb_base + u.s];
        }
      } else {
        yv = batch_f32[yb_base + u.s];
      }
      od_val = od_val + yv * w * er_v;
    }
  }
  od_buf[a * u.s + t] = od_val;
}
`;

export const ACC_HOME_WGSL = /* wgsl */ `
${UNIFORMS_WGSL}
${MATH_WGSL}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> net_f32: array<f32>;
@group(0) @binding(2) var<storage, read> net_u32: array<u32>;
@group(0) @binding(3) var<storage, read> er_buf: array<f32>;
@group(0) @binding(4) var<storage, read> batch_f32: array<f32>;
@group(0) @binding(5) var<storage, read> batch_u32: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_scalars: array<f32>;

var<workgroup> sh_sum: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let u_local = wid.y;
  let tid = lid.x;

  let bm = u.off_batch_meta_u32 + u.batch_idx * 4u;
  let u_start = net_u32[bm + 0u];
  let u_end = net_u32[bm + 1u];
  let u_ord = u_start + u_local;
  if (u_ord >= u_end) {
    return;
  }

  let mu_base = u.off_u_meta_u32 + u_ord * 8u;
  let walk_start = net_u32[mu_base + 2u];
  let walk_end = net_u32[mu_base + 3u];
  let ride_end = net_u32[mu_base + 4u];
  let rides = ride_end > walk_end;
  let used = batch_u32[u_local];
  let ss = u.s * u.s;
  let aby_off = bf_off_ab_y(u);

  for (var k = walk_start + wid.x; k < ride_end; k = k + nwg.x) {
    let is_walk = k < walk_end;
    let a = net_u32[u.off_acc_st + k];

    var acc: f32 = 0.0;
    for (var b = 0u; b < u.nb; b = b + 1u) {
      if (((used >> b) & 1u) == 0u) {
        continue;
      }
      let w = net_f32[u.off_acc_e + b * u.na_entries + k];
      let c_ub = select(0.0, net_f32[u.off_u_c_u + u_ord * u.nb + b], rides && is_walk);
      let er_row = b * ss + a * u.s;
      let y_row = aby_off + ((u_local * u.nb + b) * 2u) * u.s;

      var band_t: f32 = 0.0;
      for (var t = tid; t < u.s; t = t + 64u) {
        var yv: f32;
        if (is_walk) {
          yv = batch_f32[y_row + t] + select(0.0, c_ub * batch_f32[y_row + u.s + t], rides);
        } else {
          yv = batch_f32[y_row + u.s + t];
        }
        band_t = band_t + yv * w * er_buf[er_row + t];
      }
      acc = acc + band_t;
    }

    sh_sum[tid] = acc;
    workgroupBarrier();
    if (tid < 32u) { sh_sum[tid] = sh_sum[tid] + sh_sum[tid + 32u]; }
    workgroupBarrier();
    if (tid < 16u) { sh_sum[tid] = sh_sum[tid] + sh_sum[tid + 16u]; }
    workgroupBarrier();
    if (tid < 8u) { sh_sum[tid] = sh_sum[tid] + sh_sum[tid + 8u]; }
    workgroupBarrier();
    if (tid < 4u) { sh_sum[tid] = sh_sum[tid] + sh_sum[tid + 4u]; }
    workgroupBarrier();
    if (tid < 2u) { sh_sum[tid] = sh_sum[tid] + sh_sum[tid + 2u]; }
    workgroupBarrier();
    if (tid == 0u) {
      out_scalars[out_off_acc_home(u) + k] = out_scalars[out_off_acc_home(u) + k] + (sh_sum[0] + sh_sum[1]);
    }
    workgroupBarrier();
  }
}
`;

export const REDUCE_U_WGSL = /* wgsl */ `
${UNIFORMS_WGSL}
${MATH_WGSL}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> net_u32: array<u32>;
@group(0) @binding(2) var<storage, read> batch_f32: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_scalars: array<f32>;

var<workgroup> sh_vals: array<f32, 384>; // 64 * 6

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let u_local = wid.x;
  let tid = lid.x;
  let bm = u.off_batch_meta_u32 + u.batch_idx * 4u;
  let u_start = net_u32[bm + 0u];
  let u_end = net_u32[bm + 1u];
  let u_ord = u_start + u_local;
  if (u_ord >= u_end) {
    return;
  }

  let uq_u_base = bf_off_uq(u) + u_local * u.nq * 8u;
  var r: f32 = 0.0;
  var w: f32 = 0.0;
  var un: f32 = 0.0;
  var xd: f32 = 0.0;
  var dk: f32 = 0.0;
  var rk: f32 = 0.0;
  for (var q = tid; q < u.nq; q = q + 64u) {
    let p = uq_u_base + q * 8u;
    w = w + batch_f32[p + 1u];
    un = un + batch_f32[p + 2u];
    r = r + batch_f32[p + 3u];
    xd = xd + batch_f32[p + 4u];
    dk = dk + batch_f32[p + 5u];
    rk = rk + batch_f32[p + 6u];
  }
  let base = tid * 6u;
  sh_vals[base + 0u] = r;
  sh_vals[base + 1u] = w;
  sh_vals[base + 2u] = un;
  sh_vals[base + 3u] = xd;
  sh_vals[base + 4u] = dk;
  sh_vals[base + 5u] = rk;
  workgroupBarrier();

  for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
    if (tid < stride) {
      let o = (tid + stride) * 6u;
      for (var c = 0u; c < 6u; c = c + 1u) {
        sh_vals[base + c] = sh_vals[base + c] + sh_vals[o + c];
      }
    }
    workgroupBarrier();
  }

  if (tid == 0u) {
    out_scalars[out_off_rail_by_sub(u) + u_ord] = sh_vals[0];
    out_scalars[out_off_walk_lost(u) + u_ord] = sh_vals[1];
    out_scalars[out_off_unr_lost(u) + u_ord] = sh_vals[2];
    let sb = out_off_u_sense(u) + u_ord * 3u;
    out_scalars[sb + 0u] = sh_vals[3];
    out_scalars[sb + 1u] = sh_vals[4];
    out_scalars[sb + 2u] = sh_vals[5];
  }
}
`;

export const REDUCE_Q_WGSL = /* wgsl */ `
${UNIFORMS_WGSL}
${MATH_WGSL}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> net_f32: array<f32>;
@group(0) @binding(2) var<storage, read> net_u32: array<u32>;
@group(0) @binding(3) var<storage, read> batch_f32: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_scalars: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x;
  if (q >= u.nq) {
    return;
  }

  let bm = u.off_batch_meta_u32 + u.batch_idx * 4u;
  let u_start = net_u32[bm + 0u];
  let u_end = net_u32[bm + 1u];
  let batch_nu = u_end - u_start;

  let ne = net_u32[u.off_eg_n + q];
  var acc_kk: array<f32, 8>;
  let aq_base = out_off_acc_q(u) + q * KE;
  for (var kk = 0u; kk < ne; kk = kk + 1u) {
    acc_kk[kk] = out_scalars[aq_base + kk];
  }
  var lost = out_scalars[out_off_lost_q(u) + q];
  var unr = out_scalars[out_off_unr_q(u) + q];

  let ab_off = bf_off_ab_row();
  let uq_off = bf_off_uq(u);

  for (var ul = 0u; ul < batch_nu; ul = ul + 1u) {
    let p = uq_off + (ul * u.nq + q) * 8u;
    lost = lost + batch_f32[p + 1u];
    unr = unr + batch_f32[p + 2u];
    let g = batch_f32[p + 0u];
    if (g > 0.0) {
      let meta_u = bitcast<u32>(batch_f32[p + 7u]);
      let whole = ((meta_u >> 8u) & 1u) != 0u;
      if (!whole) {
        let b = meta_u & 0xfu;
        let rr = (meta_u >> 4u) & 1u;
        let ab_base = ab_off + ((ul * u.nb + b) * 2u + rr) * u.s;
        for (var kk = 0u; kk < ne; kk = kk + 1u) {
          let t = net_u32[u.off_eg_st + q * KE + kk];
          let e = net_f32[u.off_eg_e + (b * u.nq + q) * KE + kk];
          let r = batch_f32[ab_base + t];
          acc_kk[kk] = acc_kk[kk] + g * e * r;
        }
      }
    }
  }

  for (var kk = 0u; kk < ne; kk = kk + 1u) {
    out_scalars[aq_base + kk] = acc_kk[kk];
  }
  out_scalars[out_off_lost_q(u) + q] = lost;
  out_scalars[out_off_unr_q(u) + q] = unr;
}
`;

export const REDUCE_OZ_LEN_WGSL = /* wgsl */ `
${UNIFORMS_WGSL}
${MATH_WGSL}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> net_u32: array<u32>;
@group(0) @binding(2) var<storage, read> per_u32: array<u32>;
@group(0) @binding(3) var<storage, read_write> batch_f32: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let v_local = gid.x;
  let v = u.batch_v_start + v_local;
  if (v >= u.batch_v_end) {
    return;
  }

  let bm = u.off_batch_meta_u32 + u.batch_idx * 4u;
  let u_start = net_u32[bm + 0u];
  let u_end = net_u32[bm + 1u];
  let batch_nu = u_end - u_start;

  let zk = per_u32[v * 4u + 0u];
  let oi = per_u32[v * 4u + 2u];
  let q0 = net_u32[u.off_dz_q_range + 2u * zk];
  let q1 = net_u32[u.off_dz_q_range + 2u * zk + 1u];

  var r: f32 = 0.0;
  var w: f32 = 0.0;
  var un: f32 = 0.0;
  let uq_off = bf_off_uq(u);

  for (var ul = 0u; ul < batch_nu; ul = ul + 1u) {
    let u_ord = u_start + ul;
    if (net_u32[u.off_u_meta_u32 + u_ord * 8u + 1u] != oi) {
      continue;
    }
    let u_base = uq_off + ul * u.nq * 8u;
    for (var q = q0; q < q1; q = q + 1u) {
      let p = u_base + q * 8u;
      w = w + batch_f32[p + 1u];
      un = un + batch_f32[p + 2u];
      r = r + batch_f32[p + 3u];
    }
  }

  let lv_base = bf_off_len_visit(u) + v_local * 3u;
  batch_f32[lv_base + 0u] = r;
  batch_f32[lv_base + 1u] = w;
  batch_f32[lv_base + 2u] = un;
}
`;

export const LEN_PHASE1_WGSL = /* wgsl */ `
${UNIFORMS_WGSL}
${MATH_WGSL}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> net_f32: array<f32>;
@group(0) @binding(2) var<storage, read> net_u32: array<u32>;
@group(0) @binding(3) var<storage, read> per_u32: array<u32>;
@group(0) @binding(4) var<storage, read_write> batch_f32: array<f32>;

fn len_edge(b: u32) -> f32 {
  switch (b) {
    case 0u: { return 0.25; }
    case 1u: { return 0.5; }
    case 2u: { return 0.75; }
    case 3u: { return 1.0; }
    case 4u: { return 1.25; }
    case 5u: { return 1.5; }
    case 6u: { return 1.75; }
    case 7u: { return 2.0; }
    case 8u: { return 2.25; }
    case 9u: { return 2.5; }
    case 10u: { return 2.75; }
    case 11u: { return 3.0; }
    case 12u: { return 3.5; }
    case 13u: { return 4.0; }
    case 14u: { return 4.5; }
    case 15u: { return 5.0; }
    case 16u: { return 5.5; }
    case 17u: { return 6.0; }
    case 18u: { return 6.5; }
    case 19u: { return 7.0; }
    case 20u: { return 7.5; }
    case 21u: { return 8.0; }
    case 22u: { return 8.5; }
    case 23u: { return 9.0; }
    case 24u: { return 9.5; }
    case 25u: { return 10.0; }
    case 26u: { return 11.0; }
    case 27u: { return 12.0; }
    case 28u: { return 13.0; }
    case 29u: { return 14.0; }
    case 30u: { return 15.0; }
    case 31u: { return 16.0; }
    case 32u: { return 17.0; }
    case 33u: { return 18.0; }
    case 34u: { return 19.0; }
    case 35u: { return 20.0; }
    case 36u: { return 22.5; }
    case 37u: { return 25.0; }
    case 38u: { return 27.5; }
    case 39u: { return 30.0; }
    case 40u: { return 32.5; }
    case 41u: { return 35.0; }
    case 42u: { return 37.5; }
    case 43u: { return 40.0; }
    case 44u: { return 42.5; }
    case 45u: { return 45.0; }
    case 46u: { return 47.5; }
    case 47u: { return 50.0; }
    case 48u: { return 60.0; }
    case 49u: { return 70.0; }
    case 50u: { return 80.0; }
    case 51u: { return 90.0; }
    case 52u: { return 100.0; }
    case 53u: { return 125.0; }
    case 54u: { return 150.0; }
    case 55u: { return 200.0; }
    default: { return 300.0; }
  }
}

fn len_bin(km: f32) -> u32 {
  var l: u32 = 0u;
  var r: u32 = N_LEN - 1u;
  loop {
    if (l >= r) {
      break;
    }
    let m = (l + r) >> 1u;
    if (len_edge(m) <= km) {
      l = m + 1u;
    } else {
      r = m;
    }
  }
  return l;
}

fn len_mid(b: u32) -> f32 {
  let lo = select(0.0, len_edge(b - 1u), b > 0u);
  if (b < N_LEN - 1u) {
    return 0.5 * (lo + len_edge(b));
  }
  return lo * (4.0 / 3.0);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tid = gid.x;
  if (tid >= W_LEN) {
    return;
  }

  let lp_base = bf_off_len_part(u) + tid * 12u * N_LEN;
  let lv_off = bf_off_len_visit(u);
  let v_cnt = u.batch_v_end - u.batch_v_start;
  let n_side = u32(SPREAD_HALF + 1);

  for (var v_local = tid; v_local < v_cnt; v_local = v_local + W_LEN) {
    let r = batch_f32[lv_off + v_local * 3u + 0u];
    let w = batch_f32[lv_off + v_local * 3u + 1u];
    let un = batch_f32[lv_off + v_local * 3u + 2u];
    if (r == 0.0 && w == 0.0 && un == 0.0) {
      continue;
    }

    let v = u.batch_v_start + v_local;
    let meta_bits = per_u32[v * 4u + 1u];
    let hdx_hdy = per_u32[v * 4u + 3u];
    let hdx = (i32(hdx_hdy & 0xffffu) << 16u) >> 16u;
    let hdy = bitcast<i32>(hdx_hdy) >> 16u;
    let ax0 = abs(hdx);
    let ay0 = abs(hdy);
    let ax = max(ax0, ay0);
    let ay = min(ax0, ay0);

    if (ax <= SPREAD_HALF) {
      let sides = (meta_bits >> 9u) & 3u;
      let i = (sides * n_side + u32(ax)) * n_side + u32(ay);
      let e0 = net_u32[u.off_lk_off + i];
      let e1 = net_u32[u.off_lk_off + i + 1u];
      for (var ei = e0; ei < e1; ei = ei + 1u) {
        let bin = net_u32[u.off_lk_ent_bin + ei];
        let m0 = net_f32[u.off_lk_ent_f32 + ei * 2u + 0u];
        let m1 = net_f32[u.off_lk_ent_f32 + ei * 2u + 1u];
        if (r != 0.0) {
          let p0 = lp_base + bin * 2u;
          let a0 = two_sum_add(vec2<f32>(batch_f32[p0], batch_f32[p0 + 1u]), r * m0);
          batch_f32[p0] = a0.x; batch_f32[p0 + 1u] = a0.y;
          let p1 = lp_base + (N_LEN + bin) * 2u;
          let a1 = two_sum_add(vec2<f32>(batch_f32[p1], batch_f32[p1 + 1u]), r * m1);
          batch_f32[p1] = a1.x; batch_f32[p1 + 1u] = a1.y;
        }
        if (w != 0.0) {
          let p0 = lp_base + (2u * N_LEN + bin) * 2u;
          let a0 = two_sum_add(vec2<f32>(batch_f32[p0], batch_f32[p0 + 1u]), w * m0);
          batch_f32[p0] = a0.x; batch_f32[p0 + 1u] = a0.y;
          let p1 = lp_base + (3u * N_LEN + bin) * 2u;
          let a1 = two_sum_add(vec2<f32>(batch_f32[p1], batch_f32[p1 + 1u]), w * m1);
          batch_f32[p1] = a1.x; batch_f32[p1 + 1u] = a1.y;
        }
        if (un != 0.0) {
          let p0 = lp_base + (4u * N_LEN + bin) * 2u;
          let a0 = two_sum_add(vec2<f32>(batch_f32[p0], batch_f32[p0 + 1u]), un * m0);
          batch_f32[p0] = a0.x; batch_f32[p0 + 1u] = a0.y;
          let p1 = lp_base + (5u * N_LEN + bin) * 2u;
          let a1 = two_sum_add(vec2<f32>(batch_f32[p1], batch_f32[p1 + 1u]), un * m1);
          batch_f32[p1] = a1.x; batch_f32[p1 + 1u] = a1.y;
        }
      }
    } else {
      let km = 0.5 * u.lk_zone_km * sqrt(f32(ax * ax + ay * ay));
      let b = len_bin(km);
      let m = len_mid(b);
      var b2: u32 = N_LEN;
      var m2: f32 = m;
      if (km >= m) {
        if (b + 1u < N_LEN) {
          b2 = b + 1u;
          m2 = len_mid(b2);
        }
      } else {
        if (b > 0u) {
          b2 = b - 1u;
          m2 = len_mid(b2);
        }
      }
      var f: f32 = 0.0;
      if (b2 < N_LEN && m2 != m) {
        f = clamp((km - m) / (m2 - m), 0.0, 1.0);
      }
      let w0 = 1.0 - f;
      let wk0 = w0 * km;
      if (r != 0.0) {
        let p0 = lp_base + b * 2u;
        let a0 = two_sum_add(vec2<f32>(batch_f32[p0], batch_f32[p0 + 1u]), r * w0);
        batch_f32[p0] = a0.x; batch_f32[p0 + 1u] = a0.y;
        let p1 = lp_base + (N_LEN + b) * 2u;
        let a1 = two_sum_add(vec2<f32>(batch_f32[p1], batch_f32[p1 + 1u]), r * wk0);
        batch_f32[p1] = a1.x; batch_f32[p1 + 1u] = a1.y;
      }
      if (w != 0.0) {
        let p0 = lp_base + (2u * N_LEN + b) * 2u;
        let a0 = two_sum_add(vec2<f32>(batch_f32[p0], batch_f32[p0 + 1u]), w * w0);
        batch_f32[p0] = a0.x; batch_f32[p0 + 1u] = a0.y;
        let p1 = lp_base + (3u * N_LEN + b) * 2u;
        let a1 = two_sum_add(vec2<f32>(batch_f32[p1], batch_f32[p1 + 1u]), w * wk0);
        batch_f32[p1] = a1.x; batch_f32[p1 + 1u] = a1.y;
      }
      if (un != 0.0) {
        let p0 = lp_base + (4u * N_LEN + b) * 2u;
        let a0 = two_sum_add(vec2<f32>(batch_f32[p0], batch_f32[p0 + 1u]), un * w0);
        batch_f32[p0] = a0.x; batch_f32[p0 + 1u] = a0.y;
        let p1 = lp_base + (5u * N_LEN + b) * 2u;
        let a1 = two_sum_add(vec2<f32>(batch_f32[p1], batch_f32[p1 + 1u]), un * wk0);
        batch_f32[p1] = a1.x; batch_f32[p1 + 1u] = a1.y;
      }
      if (f > 0.0 && b2 < N_LEN) {
        let fk = f * km;
        if (r != 0.0) {
          let p0 = lp_base + b2 * 2u;
          let a0 = two_sum_add(vec2<f32>(batch_f32[p0], batch_f32[p0 + 1u]), r * f);
          batch_f32[p0] = a0.x; batch_f32[p0 + 1u] = a0.y;
          let p1 = lp_base + (N_LEN + b2) * 2u;
          let a1 = two_sum_add(vec2<f32>(batch_f32[p1], batch_f32[p1 + 1u]), r * fk);
          batch_f32[p1] = a1.x; batch_f32[p1 + 1u] = a1.y;
        }
        if (w != 0.0) {
          let p0 = lp_base + (2u * N_LEN + b2) * 2u;
          let a0 = two_sum_add(vec2<f32>(batch_f32[p0], batch_f32[p0 + 1u]), w * f);
          batch_f32[p0] = a0.x; batch_f32[p0 + 1u] = a0.y;
          let p1 = lp_base + (3u * N_LEN + b2) * 2u;
          let a1 = two_sum_add(vec2<f32>(batch_f32[p1], batch_f32[p1 + 1u]), w * fk);
          batch_f32[p1] = a1.x; batch_f32[p1 + 1u] = a1.y;
        }
        if (un != 0.0) {
          let p0 = lp_base + (4u * N_LEN + b2) * 2u;
          let a0 = two_sum_add(vec2<f32>(batch_f32[p0], batch_f32[p0 + 1u]), un * f);
          batch_f32[p0] = a0.x; batch_f32[p0 + 1u] = a0.y;
          let p1 = lp_base + (5u * N_LEN + b2) * 2u;
          let a1 = two_sum_add(vec2<f32>(batch_f32[p1], batch_f32[p1 + 1u]), un * fk);
          batch_f32[p1] = a1.x; batch_f32[p1 + 1u] = a1.y;
        }
      }
    }
  }
}
`;

export const LEN_PHASE2_WGSL = /* wgsl */ `
${UNIFORMS_WGSL}
${MATH_WGSL}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> batch_f32: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_scalars: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let slot = gid.x;
  if (slot >= 6u * N_LEN) {
    return;
  }
  var acc = vec2<f32>(0.0, 0.0);
  let lp_off = bf_off_len_part(u) + slot * 2u;
  for (var tid = 0u; tid < W_LEN; tid = tid + 1u) {
    let p = lp_off + tid * 12u * N_LEN;
    let hi = batch_f32[p];
    let lo = batch_f32[p + 1u];
    let r = two_sum_add(acc, hi);
    acc = vec2<f32>(r.x, r.y + lo);
  }
  out_scalars[out_off_len(u) + slot] = acc.x + acc.y;
}
`;

export const SLOW_BIN_WGSL = /* wgsl */ `
${UNIFORMS_WGSL}
${MATH_WGSL}

struct SlowBatchU32 {
  hdr: array<atomic<u32>, 324>,
  data: array<u32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> batch_u32: SlowBatchU32;
@group(0) @binding(2) var<storage, read_write> out_scalars: array<f32>;

var<workgroup> sh_slow_cnt: u32;
var<workgroup> sh_ovf: u32;
var<workgroup> sh_lo: u32;
var<workgroup> sh_hi: u32;

@compute @workgroup_size(64, 1, 1)
fn slow_prep_u(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  if (lid.x == 0u) {
    sh_slow_cnt = atomicLoad(&batch_u32.hdr[128u]);
    sh_ovf = atomicLoad(&batch_u32.hdr[129u]);
  }
  let slow_cnt = workgroupUniformLoad(&sh_slow_cnt);
  let ovf = workgroupUniformLoad(&sh_ovf);
  if (slow_cnt == 0u) {
    return;
  }
  if (slow_cnt > MAX_SLOW || ovf != 0u) {
    if (gid.x == 0u) {
      atomicStore(&batch_u32.hdr[129u], 1u);
      out_scalars[out_off_status(u)] = 1.0;
      out_scalars[out_off_status(u) + 1u] = max(out_scalars[out_off_status(u) + 1u], f32(slow_cnt));
    }
    return;
  }

  let tid = lid.x;
  let recs_base = bu_off_slow_recs(u) - BATCH_U32_HDR;
  for (var idx = tid; idx < slow_cnt; idx = idx + 64u) {
    let key0 = batch_u32.data[recs_base + idx * 8u + 0u];
    let u_l = key0 >> 24u;
    atomicAdd(&batch_u32.hdr[130u + u_l], 1u);
  }
  workgroupBarrier();

  if (tid == 0u) {
    var sum: u32 = 0u;
    for (var i = 0u; i < 64u; i = i + 1u) {
      atomicStore(&batch_u32.hdr[194u + i], sum);
      atomicStore(&batch_u32.hdr[260u + i], sum);
      sum = sum + atomicLoad(&batch_u32.hdr[130u + i]);
    }
    atomicStore(&batch_u32.hdr[194u + 64u], sum);
  }
}

@compute @workgroup_size(256, 1, 1)
fn slow_scatter_u(@builtin(global_invocation_id) gid: vec3<u32>) {
  let slow_cnt = atomicLoad(&batch_u32.hdr[128u]);
  if (slow_cnt == 0u || slow_cnt > MAX_SLOW || atomicLoad(&batch_u32.hdr[129u]) != 0u) {
    return;
  }
  let recs_base = bu_off_slow_recs(u) - BATCH_U32_HDR;
  let sorted_base = bu_off_slow_sorted(u) - BATCH_U32_HDR;
  for (var idx = gid.x; idx < slow_cnt; idx = idx + 256u) {
    let key0 = batch_u32.data[recs_base + idx * 8u + 0u];
    let u_l = key0 >> 24u;
    let dst = atomicAdd(&batch_u32.hdr[260u + u_l], 1u);
    batch_u32.data[sorted_base + dst] = idx;
  }
}

var<workgroup> sh_key0: array<u32, 1024>;
var<workgroup> sh_key1: array<u32, 1024>;
var<workgroup> sh_rec: array<u32, 1024>;

fn key_gt(k0_a: u32, k1_a: u32, k0_b: u32, k1_b: u32) -> bool {
  return (k0_a > k0_b) || (k0_a == k0_b && k1_a > k1_b);
}

@compute @workgroup_size(256, 1, 1)
fn slow_sort_u(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let u_l = wid.x;
  if (lid.x == 0u) {
    sh_slow_cnt = atomicLoad(&batch_u32.hdr[128u]);
    sh_ovf = atomicLoad(&batch_u32.hdr[129u]);
    sh_lo = atomicLoad(&batch_u32.hdr[194u + u_l]);
    sh_hi = atomicLoad(&batch_u32.hdr[194u + u_l + 1u]);
  }
  let slow_cnt = workgroupUniformLoad(&sh_slow_cnt);
  let ovf = workgroupUniformLoad(&sh_ovf);
  if (slow_cnt == 0u || slow_cnt > MAX_SLOW || ovf != 0u) {
    return;
  }
  let lo = workgroupUniformLoad(&sh_lo);
  let hi = workgroupUniformLoad(&sh_hi);
  let m = hi - lo;
  if (m == 0u) {
    return;
  }

  let tid = lid.x;
  let recs_base = bu_off_slow_recs(u) - BATCH_U32_HDR;
  let sorted_base = bu_off_slow_sorted(u) - BATCH_U32_HDR;
  let compact_base = bu_off_slow_compact(u) - BATCH_U32_HDR;

  if (m == 1u) {
    if (tid == 0u) {
      let rec_idx = batch_u32.data[sorted_base + lo];
      let rec = recs_base + rec_idx * 8u;
      let c = compact_base + lo * 4u;
      batch_u32.data[c + 0u] = batch_u32.data[rec + 2u] | (batch_u32.data[rec + 3u] << 16u);
      batch_u32.data[c + 1u] = batch_u32.data[rec + 4u] | (batch_u32.data[rec + 6u] << 24u);
      batch_u32.data[c + 2u] = batch_u32.data[rec + 5u];
      batch_u32.data[c + 3u] = batch_u32.data[rec + 7u];
    }
    return;
  }

  if (m <= 1024u) {
    var p: u32 = 1u;
    loop {
      if (p >= m) { break; }
      p = p << 1u;
    }
    for (var i = tid; i < p; i = i + 256u) {
      if (i < m) {
        let rec_idx = batch_u32.data[sorted_base + lo + i];
        sh_key0[i] = batch_u32.data[recs_base + rec_idx * 8u + 0u];
        sh_key1[i] = batch_u32.data[recs_base + rec_idx * 8u + 1u];
        sh_rec[i] = rec_idx;
      } else {
        sh_key0[i] = 0xffffffffu;
        sh_key1[i] = 0xffffffffu;
        sh_rec[i] = 0u;
      }
    }
    workgroupBarrier();

    var k: u32 = 2u;
    loop {
      if (k > p) { break; }
      var j: u32 = k >> 1u;
      loop {
        if (j == 0u) { break; }
        for (var i = tid; i < p; i = i + 256u) {
          let ixj = i ^ j;
          if (ixj > i) {
            let asc = (i & k) == 0u;
            let k0i = sh_key0[i]; let k1i = sh_key1[i]; let ri = sh_rec[i];
            let k0x = sh_key0[ixj]; let k1x = sh_key1[ixj]; let rx = sh_rec[ixj];
            let swap = select(key_gt(k0x, k1x, k0i, k1i), key_gt(k0i, k1i, k0x, k1x), asc);
            if (swap) {
              sh_key0[i] = k0x; sh_key1[i] = k1x; sh_rec[i] = rx;
              sh_key0[ixj] = k0i; sh_key1[ixj] = k1i; sh_rec[ixj] = ri;
            }
          }
        }
        workgroupBarrier();
        j = j >> 1u;
      }
      k = k << 1u;
    }

    for (var i = tid; i < m; i = i + 256u) {
      let rec_idx = sh_rec[i];
      batch_u32.data[sorted_base + lo + i] = rec_idx;
      let rec = recs_base + rec_idx * 8u;
      let c = compact_base + (lo + i) * 4u;
      batch_u32.data[c + 0u] = batch_u32.data[rec + 2u] | (batch_u32.data[rec + 3u] << 16u);
      batch_u32.data[c + 1u] = batch_u32.data[rec + 4u] | (batch_u32.data[rec + 6u] << 24u);
      batch_u32.data[c + 2u] = batch_u32.data[rec + 5u];
      batch_u32.data[c + 3u] = batch_u32.data[rec + 7u];
    }
    return;
  }

  // Fallback for rare huge u_local (> 1024 records): in-place odd-even transposition sort
  for (var phase = 0u; phase < m; phase = phase + 1u) {
    let start = phase & 1u;
    for (var i = start + tid * 2u; i + 1u < m; i = i + 512u) {
      let r0 = batch_u32.data[sorted_base + lo + i];
      let r1 = batch_u32.data[sorted_base + lo + i + 1u];
      let k0a = batch_u32.data[recs_base + r0 * 8u + 0u];
      let k1a = batch_u32.data[recs_base + r0 * 8u + 1u];
      let k0b = batch_u32.data[recs_base + r1 * 8u + 0u];
      let k1b = batch_u32.data[recs_base + r1 * 8u + 1u];
      if (key_gt(k0a, k1a, k0b, k1b)) {
        batch_u32.data[sorted_base + lo + i] = r1;
        batch_u32.data[sorted_base + lo + i + 1u] = r0;
      }
    }
    storageBarrier();
  }
  for (var i = tid; i < m; i = i + 256u) {
    let rec_idx = batch_u32.data[sorted_base + lo + i];
    let rec = recs_base + rec_idx * 8u;
    let c = compact_base + (lo + i) * 4u;
    batch_u32.data[c + 0u] = batch_u32.data[rec + 2u] | (batch_u32.data[rec + 3u] << 16u);
    batch_u32.data[c + 1u] = batch_u32.data[rec + 4u] | (batch_u32.data[rec + 6u] << 24u);
    batch_u32.data[c + 2u] = batch_u32.data[rec + 5u];
    batch_u32.data[c + 3u] = batch_u32.data[rec + 7u];
  }
}

var<workgroup> sh_cnt: array<u32, 2048>; // 64 buckets * 32 threads
var<workgroup> sh_tot: array<u32, 64>;
var<workgroup> sh_scan: array<u32, 256>;

@compute @workgroup_size(32, 1, 1)
fn slow_bkt_count(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  if (lid.x == 0u) {
    sh_slow_cnt = atomicLoad(&batch_u32.hdr[128u]);
    sh_ovf = atomicLoad(&batch_u32.hdr[129u]);
  }
  let slow_cnt = workgroupUniformLoad(&sh_slow_cnt);
  let ovf = workgroupUniformLoad(&sh_ovf);
  if (slow_cnt == 0u || slow_cnt > MAX_SLOW || ovf != 0u) {
    return;
  }

  let t = wid.x * 32u + lid.x; // 0..255
  let tgt = wid.y;          // 0=od, 1=q, 2=home
  for (var b = 0u; b < 64u; b = b + 1u) {
    sh_cnt[b * 32u + lid.x] = 0u;
  }

  let step = (slow_cnt + 255u) / 256u;
  let c_lo = min(t * step, slow_cnt);
  let c_hi = min(c_lo + step, slow_cnt);
  let compact_base = bu_off_slow_compact(u) - BATCH_U32_HDR;

  for (var i = c_lo; i < c_hi; i = i + 1u) {
    let w_key = batch_u32.data[compact_base + i * 4u + tgt];
    let b = w_key & 63u;
    sh_cnt[b * 32u + lid.x] = sh_cnt[b * 32u + lid.x] + 1u;
  }

  let cnt_base = (bu_off_slow_bkt_cnt(u) - BATCH_U32_HDR) + tgt * 16384u;
  for (var b = 0u; b < 64u; b = b + 1u) {
    batch_u32.data[cnt_base + b * 256u + t] = sh_cnt[b * 32u + lid.x];
  }
}

@compute @workgroup_size(256, 1, 1)
fn slow_bkt_prefix1(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  if (lid.x == 0u) {
    sh_slow_cnt = atomicLoad(&batch_u32.hdr[128u]);
    sh_ovf = atomicLoad(&batch_u32.hdr[129u]);
  }
  let slow_cnt = workgroupUniformLoad(&sh_slow_cnt);
  let ovf = workgroupUniformLoad(&sh_ovf);
  if (slow_cnt == 0u || slow_cnt > MAX_SLOW || ovf != 0u) {
    return;
  }

  let b = wid.x;
  let tgt = wid.y;
  let t = lid.x;
  let idx = (bu_off_slow_bkt_cnt(u) - BATCH_U32_HDR) + tgt * 16384u + b * 256u + t;
  let val = batch_u32.data[idx];
  sh_scan[t] = val;
  workgroupBarrier();

  for (var off = 1u; off < 256u; off = off << 1u) {
    var add_v: u32 = 0u;
    if (t >= off) {
      add_v = sh_scan[t - off];
    }
    workgroupBarrier();
    if (t >= off) {
      sh_scan[t] = sh_scan[t] + add_v;
    }
    workgroupBarrier();
  }

  batch_u32.data[idx] = sh_scan[t] - val;
  if (t == 255u) {
    let bounds_base = (bu_off_slow_bkt_bounds(u) - BATCH_U32_HDR) + tgt * 65u;
    batch_u32.data[bounds_base + b] = sh_scan[255u];
  }
}

@compute @workgroup_size(64, 1, 1)
fn slow_bkt_prefix2(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  if (lid.x == 0u) {
    sh_slow_cnt = atomicLoad(&batch_u32.hdr[128u]);
    sh_ovf = atomicLoad(&batch_u32.hdr[129u]);
  }
  let slow_cnt = workgroupUniformLoad(&sh_slow_cnt);
  let ovf = workgroupUniformLoad(&sh_ovf);
  if (slow_cnt == 0u || slow_cnt > MAX_SLOW || ovf != 0u) {
    return;
  }

  let b = lid.x;
  let tgt = wid.y;
  let bounds_base = (bu_off_slow_bkt_bounds(u) - BATCH_U32_HDR) + tgt * 65u;
  let sum_b = batch_u32.data[bounds_base + b];
  sh_tot[b] = sum_b;
  workgroupBarrier();

  for (var off = 1u; off < 64u; off = off << 1u) {
    var add_v: u32 = 0u;
    if (b >= off) {
      add_v = sh_tot[b - off];
    }
    workgroupBarrier();
    if (b >= off) {
      sh_tot[b] = sh_tot[b] + add_v;
    }
    workgroupBarrier();
  }

  let start_b = sh_tot[b] - sum_b;
  batch_u32.data[bounds_base + b] = start_b;
  if (b == 63u) {
    batch_u32.data[bounds_base + 64u] = start_b + sum_b;
  }
}

@compute @workgroup_size(32, 1, 1)
fn slow_bkt_scatter(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  if (lid.x == 0u) {
    sh_slow_cnt = atomicLoad(&batch_u32.hdr[128u]);
    sh_ovf = atomicLoad(&batch_u32.hdr[129u]);
  }
  let slow_cnt = workgroupUniformLoad(&sh_slow_cnt);
  let ovf = workgroupUniformLoad(&sh_ovf);
  if (slow_cnt == 0u || slow_cnt > MAX_SLOW || ovf != 0u) {
    return;
  }

  let t = wid.x * 32u + lid.x;
  let tgt = wid.y;
  let cnt_base = (bu_off_slow_bkt_cnt(u) - BATCH_U32_HDR) + tgt * 16384u;
  let bounds_base = (bu_off_slow_bkt_bounds(u) - BATCH_U32_HDR) + tgt * 65u;
  for (var b = 0u; b < 64u; b = b + 1u) {
    sh_cnt[b * 32u + lid.x] = batch_u32.data[bounds_base + b] + batch_u32.data[cnt_base + b * 256u + t];
  }

  let step = (slow_cnt + 255u) / 256u;
  let c_lo = min(t * step, slow_cnt);
  let c_hi = min(c_lo + step, slow_cnt);
  let compact_base = bu_off_slow_compact(u) - BATCH_U32_HDR;
  let bkt_base = (bu_off_slow_bkt_data(u) - BATCH_U32_HDR) + tgt * MAX_SLOW * 2u;

  for (var i = c_lo; i < c_hi; i = i + 1u) {
    let c = compact_base + i * 4u;
    let w_key = batch_u32.data[c + tgt];
    let w_f = batch_u32.data[c + 3u];
    let b = w_key & 63u;
    let dst = sh_cnt[b * 32u + lid.x];
    sh_cnt[b * 32u + lid.x] = dst + 1u;
    batch_u32.data[bkt_base + dst * 2u + 0u] = w_key;
    batch_u32.data[bkt_base + dst * 2u + 1u] = w_f;
  }
}
`;

export const SLOW_APPLY_OD_WGSL = /* wgsl */ `
${UNIFORMS_WGSL}
${MATH_WGSL}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> batch_u32: array<u32>;
@group(0) @binding(2) var<storage, read_write> od_buf: array<f32>;

var<workgroup> sh_slow_cnt: u32;
var<workgroup> sh_ovf: u32;
var<workgroup> sh_lo: u32;
var<workgroup> sh_hi: u32;
var<workgroup> sh_w0: array<u32, 64>;
var<workgroup> sh_w3: array<u32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let b = wid.x;
  let tid = lid.x;
  if (tid == 0u) {
    sh_slow_cnt = batch_u32[128u];
    sh_ovf = batch_u32[129u];
    sh_lo = batch_u32[bu_off_slow_bkt_bounds(u) + b];
    sh_hi = batch_u32[bu_off_slow_bkt_bounds(u) + b + 1u];
  }
  let slow_cnt = workgroupUniformLoad(&sh_slow_cnt);
  let ovf = workgroupUniformLoad(&sh_ovf);
  let b_lo = workgroupUniformLoad(&sh_lo);
  let b_hi = workgroupUniformLoad(&sh_hi);
  if (slow_cnt == 0u || slow_cnt > MAX_SLOW || ovf != 0u || b_lo >= b_hi) {
    return;
  }

  let bkt_base = bu_off_slow_bkt_data(u);
  for (var base = b_lo; base < b_hi; base = base + 64u) {
    let idx = base + tid;
    if (idx < b_hi) {
      sh_w0[tid] = batch_u32[bkt_base + idx * 2u + 0u];
      sh_w3[tid] = batch_u32[bkt_base + idx * 2u + 1u];
    }
    workgroupBarrier();

    let n_tile = min(64u, b_hi - base);
    for (var j = 0u; j < n_tile; j = j + 1u) {
      let w0 = sh_w0[j];
      let a = w0 & 0xffffu;
      if (((a >> 6u) & 63u) == tid) {
        let t = w0 >> 16u;
        let f = bitcast<f32>(sh_w3[j]);
        od_buf[a * u.s + t] = od_buf[a * u.s + t] + f;
      }
    }
    workgroupBarrier();
  }
}
`;

export const SLOW_APPLY_REST_WGSL = /* wgsl */ `
${UNIFORMS_WGSL}
${MATH_WGSL}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> batch_u32: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_scalars: array<f32>;

var<workgroup> sh_slow_cnt: u32;
var<workgroup> sh_ovf: u32;
var<workgroup> sh_lo: u32;
var<workgroup> sh_hi: u32;
var<workgroup> sh_idx: array<u32, 64>;
var<workgroup> sh_w3: array<u32, 64>;

@compute @workgroup_size(64, 1, 1)
fn slow_apply_q(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let b = wid.x;
  let tid = lid.x;
  if (tid == 0u) {
    sh_slow_cnt = batch_u32[128u];
    sh_ovf = batch_u32[129u];
    let bounds = bu_off_slow_bkt_bounds(u) + 65u;
    sh_lo = batch_u32[bounds + b];
    sh_hi = batch_u32[bounds + b + 1u];
  }
  let slow_cnt = workgroupUniformLoad(&sh_slow_cnt);
  let ovf = workgroupUniformLoad(&sh_ovf);
  let b_lo = workgroupUniformLoad(&sh_lo);
  let b_hi = workgroupUniformLoad(&sh_hi);
  if (slow_cnt == 0u || slow_cnt > MAX_SLOW || ovf != 0u || b_lo >= b_hi) {
    return;
  }

  let bkt_base = bu_off_slow_bkt_data(u) + MAX_SLOW * 2u;
  let aq_off = out_off_acc_q(u);

  for (var base = b_lo; base < b_hi; base = base + 64u) {
    let idx = base + tid;
    if (idx < b_hi) {
      sh_idx[tid] = batch_u32[bkt_base + idx * 2u + 0u];
      sh_w3[tid] = batch_u32[bkt_base + idx * 2u + 1u];
    }
    workgroupBarrier();

    let n_tile = min(64u, b_hi - base);
    for (var j = 0u; j < n_tile; j = j + 1u) {
      let w1 = sh_idx[j];
      let kk = w1 >> 24u;
      let lane = (((w1 >> 6u) & 7u) << 3u) | kk;
      if (lane == tid) {
        let q = w1 & 0x00ffffffu;
        let f = bitcast<f32>(sh_w3[j]);
        let p = aq_off + q * KE + kk;
        out_scalars[p] = out_scalars[p] + f;
      }
    }
    workgroupBarrier();
  }
}

@compute @workgroup_size(64, 1, 1)
fn slow_apply_home(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let b = wid.x;
  let tid = lid.x;
  if (tid == 0u) {
    sh_slow_cnt = batch_u32[128u];
    sh_ovf = batch_u32[129u];
    let bounds = bu_off_slow_bkt_bounds(u) + 130u;
    sh_lo = batch_u32[bounds + b];
    sh_hi = batch_u32[bounds + b + 1u];
  }
  let slow_cnt = workgroupUniformLoad(&sh_slow_cnt);
  let ovf = workgroupUniformLoad(&sh_ovf);
  let b_lo = workgroupUniformLoad(&sh_lo);
  let b_hi = workgroupUniformLoad(&sh_hi);
  if (slow_cnt == 0u || slow_cnt > MAX_SLOW || ovf != 0u || b_lo >= b_hi) {
    return;
  }

  let bkt_base = bu_off_slow_bkt_data(u) + MAX_SLOW * 4u;
  let ah_off = out_off_acc_home(u);

  for (var base = b_lo; base < b_hi; base = base + 64u) {
    let idx = base + tid;
    if (idx < b_hi) {
      sh_idx[tid] = batch_u32[bkt_base + idx * 2u + 0u];
      sh_w3[tid] = batch_u32[bkt_base + idx * 2u + 1u];
    }
    workgroupBarrier();

    let n_tile = min(64u, b_hi - base);
    for (var j = 0u; j < n_tile; j = j + 1u) {
      let k = sh_idx[j];
      if (((k >> 6u) & 63u) == tid) {
        let f = bitcast<f32>(sh_w3[j]);
        let p = ah_off + k;
        out_scalars[p] = out_scalars[p] + f;
      }
    }
    workgroupBarrier();
  }
}
`;

export const REDUCE_BANDS_WGSL = /* wgsl */ `
${UNIFORMS_WGSL}
${MATH_WGSL}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> net_u32: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_scalars: array<f32>;

var<workgroup> sh_hi: array<f32, 64>;
var<workgroup> sh_lo: array<f32, 64>;

fn two_sum_pair(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  let r = two_sum_add(a, b.x);
  return vec2<f32>(r.x, r.y + b.y);
}

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let band = wid.x;
  if (band >= N_BANDS) {
    return;
  }
  let tid = lid.x;
  var acc = vec2<f32>(0.0, 0.0);
  let ah_off = out_off_acc_home(u);
  for (var k = tid; k < u.na_entries; k = k + 64u) {
    if (net_u32[u.off_acc_band + k] == band) {
      acc = two_sum_add(acc, out_scalars[ah_off + k]);
    }
  }
  sh_hi[tid] = acc.x;
  sh_lo[tid] = acc.y;
  workgroupBarrier();

  for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
    if (tid < stride) {
      let p = two_sum_pair(
        vec2<f32>(sh_hi[tid], sh_lo[tid]),
        vec2<f32>(sh_hi[tid + stride], sh_lo[tid + stride]),
      );
      sh_hi[tid] = p.x;
      sh_lo[tid] = p.y;
    }
    workgroupBarrier();
  }

  if (tid == 0u) {
    out_scalars[out_off_rail_band(u) + band] = sh_hi[0] + sh_lo[0];
  }
}
`;
