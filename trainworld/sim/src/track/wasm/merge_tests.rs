//! T-3376: a junction's merge lane, its price and the query the junction view reads.

use super::*;

fn api() -> TrackApi {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_cash(1e9);
    a
}
fn free(x: f64, y: f64) -> [f64; 5] {
    [0.0, 0.0, x, y, 0.0]
}
fn ends(a: [f64; 5], b: [f64; 5]) -> Vec<f64> {
    a.iter().chain(b.iter()).copied().collect()
}
fn alive(a: &TrackApi) -> Vec<u32> {
    (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).collect()
}
/// The node nearest (x, y) with 3+ edge ends.
fn junction_near(a: &TrackApi, x: f64, y: f64) -> u32 {
    let n = &a.w.net;
    (0..n.node_count() as u32)
        .filter(|&k| n.node_ok(k) && n.node_ports[k as usize].n >= 3)
        .min_by(|&p, &q| {
            let d = |k: u32| (n.node_x[k as usize] - x).hypot(n.node_y[k as usize] - y);
            d(p).total_cmp(&d(q))
        })
        .unwrap()
}
/// A main through (0, y)-(3000, y) and a branch leaving it at x = 1000 for (2000, y - 600).
fn y_junction(a: &mut TrackApi, y: f64) -> u32 {
    assert_eq!(a.add_route(&ends(free(0.0, y), free(3000.0, y)), &[], 2, false), 0, "{}", a.issues());
    let e = *alive(a).iter().find(|&&e| a.w.net.project(e, 1000.0, y).1 < 1.0).unwrap();
    assert_eq!(a.add_route(&ends([2.0, e as f64, 1000.0, y, 0.0], free(2000.0, y - 600.0)), &[], 2, false), 0, "{}", a.issues());
    junction_near(a, 1000.0, y)
}

#[test]
fn t3376_merge_lane_is_priced_toggled_and_saved() {
    let mut a = api();
    let j = y_junction(&mut a, 0.0);
    let before = a.merge_lane_info(j);
    assert_eq!(before.len(), 12);
    assert!((before[0] - 0.3 * 37.5).abs() < 1e-9, "0.3 km at ground: {}", before[0]);
    assert_eq!(before[1], 0.0, "not built yet");
    let quote = a.blueprint_cost();
    assert_eq!(a.set_merge_lane(j, true), 0, "{}", a.issues());
    assert!(a.w.net.node_merge_lane[j as usize]);
    assert!((a.blueprint_cost() - quote - before[0]).abs() < 1e-9, "the lane joins the quote");
    let on = a.merge_lane_info(j);
    assert_eq!((on[0], on[1]), (0.0, 1.0), "no price once it is there");
    // it rides in the save, and comes back off
    let saved = a.save();
    let mut b = api();
    assert!(b.load(&saved));
    assert!(b.w.net.node_merge_lane[j as usize]);
    assert_eq!(b.merge_lane_info(j)[1], 1.0);
    assert_eq!(a.set_merge_lane(j, false), 0);
    assert!(!a.w.net.node_merge_lane[j as usize]);
}

#[test]
fn t3376_merge_lane_guards_non_junction_built_and_crossover() {
    let mut a = api();
    let j = y_junction(&mut a, 0.0);
    // A 1-port dead-end is not a junction
    let end = (0..a.w.net.node_count() as u32)
        .find(|&n| a.w.net.node_ok(n) && a.w.net.node_ports[n as usize].n == 1)
        .unwrap();
    assert_ne!(a.set_merge_lane(end, true), 0);
    assert!(!a.w.net.node_merge_lane[end as usize]);
    assert!(a.merge_lane_info(end).iter().all(|&v| v == 0.0));

    // Construct only 2 ports of j (partial junction): still removable before construction finishes
    let mut edges: Vec<u32> = a.w.net.node_ports[j as usize].ends().iter().map(|&c| c >> 1).collect();
    edges.sort_unstable();
    assert_eq!(a.construct(&edges[..2], &[]), 0, "{}", a.issues());
    assert_eq!(a.set_merge_lane(j, true), 0, "{}", a.issues());
    assert_eq!(a.set_merge_lane(j, false), 0, "{}", a.issues());

    // Once all 3 ports are built, adding charges at once and becomes final (removal refused)
    assert_eq!(a.construct(&[edges[2]], &[]), 0, "{}", a.issues());
    let cash_before = a.cash();
    assert_eq!(a.set_merge_lane(j, true), 0, "{}", a.issues());
    assert!(cash_before - a.cash() > 1.0);
    assert_eq!(a.set_merge_lane(j, true), 0, "idempotent enable succeeds");
    assert_ne!(a.set_merge_lane(j, false), 0, "cannot remove merge lane on built junction");
    assert!(a.w.net.node_merge_lane[j as usize]);

    // A crossover node refuses set_merge_lane and merge_lane_info reports zeros
    let mut b = api();
    let gap = 12.0 * b.w.net.k_at(0.0);
    assert_eq!(b.add_route(&ends(free(0.0, 0.0), free(3000.0, 0.0)), &[], 2, false), 0, "{}", b.issues());
    assert_eq!(b.add_route(&ends(free(0.0, gap), free(3000.0, gap)), &[], 2, false), 0, "{}", b.issues());
    let x = alive(&b)[0];
    assert_eq!(b.add_crossover(x, 1200.0, 0.0), 0, "{}", b.issues());
    let cx_node = (0..b.w.net.node_count() as u32)
        .find(|&n| b.w.net.node_ok(n) && b.w.net.node_ports[n as usize].ends().iter().any(|&c| b.w.net.edge_crossover[(c >> 1) as usize]))
        .unwrap();
    assert_ne!(b.set_merge_lane(cx_node, true), 0);
    assert!(b.merge_lane_info(cx_node).iter().all(|&v| v == 0.0));
}

#[test]
fn t3376_merge_at_is_deterministic_on_dual_merge_4way() {
    let mut w = crate::track::tests::world();
    let west = crate::track::tests::node(&mut w, -3000.0, 0.0, 0);
    let j = crate::track::tests::node(&mut w, 0.0, 0.0, 0);
    let east = crate::track::tests::node(&mut w, 3000.0, 0.0, 0);
    let nw = crate::track::tests::node(&mut w, -2500.0, 1500.0, 0);
    let se = crate::track::tests::node(&mut w, 2500.0, -1500.0, 0);
    crate::track::tests::edge(&mut w, west, j, &[], 0).unwrap();
    crate::track::tests::edge(&mut w, j, east, &[], 0).unwrap();
    crate::track::tests::edge(&mut w, nw, j, &[(-400.0, 0.0)], 0).unwrap();
    crate::track::tests::edge(&mut w, j, se, &[(400.0, 0.0)], 0).unwrap();
    for n in [west, east, nw, se] {
        crate::track::tests::station(&mut w, n, 200);
    }
    let _l1 = crate::track::tests::line(&mut w, &[west, east], [20.0, 10.0, 5.0]);
    let _l2 = crate::track::tests::line(&mut w, &[nw, east], [8.0, 4.0, 2.0]);
    let _l3 = crate::track::tests::line(&mut w, &[se, west], [16.0, 8.0, 4.0]);
    // Toggle merge_lane incrementally, then compare merge_at peak rho against global solve
    let mut d = w.net.node_data(j).unwrap();
    d.merge_lane = true;
    w.apply(Op::Node { id: j, data: Some(d.clone()) }).unwrap();
    d.merge_lane = false;
    w.apply(Op::Node { id: j, data: Some(d) }).unwrap();
    let mut global = super::super::capacity::build(&w.net, &w.svc.lines);
    super::super::capacity::solve(&mut global, &w.net);
    let r_inc = w.svc.cap.merge_at(j).expect("incremental merge");
    let r_glo = global.merge_at(j).expect("global merge");
    assert_eq!(w.svc.cap.rho[r_inc], global.rho[r_glo]);
}

#[test]
fn t3341_partial_line_upkeep_ignores_unserved_built_station() {
    let mut a = api();
    let mut w = crate::track::tests::world();
    let s0 = crate::track::tests::node(&mut w, 0.0, 0.0, 0);
    let s1 = crate::track::tests::node(&mut w, 3000.0, 0.0, 0);
    let s2 = crate::track::tests::node(&mut w, 6000.0, 0.0, 0);
    crate::track::tests::edge(&mut w, s0, s1, &[], 0).unwrap();
    crate::track::tests::planned_edge(&mut w, s1, s2, &[], 0);
    for n in [s0, s1, s2] {
        crate::track::tests::station(&mut w, n, 200);
    }
    let _l = crate::track::tests::line(&mut w, &[s0, s1, s2], [12.0, 6.0, 3.0]);
    a.w = w;
    let up = a.line_upkeep(&[4.0, 8.0, 12.0]);
    let rest = up.chunks(2).find(|c| c[0] == -1.0).map(|c| c[1]).unwrap_or(0.0);
    assert!(rest > 0.0, "unserved built station upkeep belongs to unserved rest, got {up:?}");
}
