// Headless Chromium WebGPU parity, determinism, worker, and fallback checks for mode choice.
import assert from 'node:assert/strict';
import http from 'node:http';
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync } from 'node:fs';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { playwright } from './lib.mjs';

const HERE = fileURLToPath(new URL('.', import.meta.url));

async function loadPlaywright() {
  try {
    return await playwright();
  } catch {
    const npx = join(process.env.HOME ?? '', '.npm', '_npx');
    for (const d of existsSync(npx) ? readdirSync(npx) : []) {
      const p = join(npx, d, 'node_modules', 'playwright', 'index.mjs');
      if (existsSync(p)) return import(pathToFileURL(p).href);
    }
    throw new Error('Playwright not found');
  }
}

function findChromeExecutable() {
  if (process.env.CHROME_BIN && existsSync(process.env.CHROME_BIN)) {
    return process.env.CHROME_BIN;
  }
  if (process.platform === 'linux') {
    const cacheDir = join(process.env.HOME ?? '', '.cache', 'ms-playwright');
    if (existsSync(cacheDir)) {
      const dirs = readdirSync(cacheDir)
        .filter((d) => d.startsWith('chromium-'))
        .sort()
        .reverse();
      for (const d of dirs) {
        const candidate = join(cacheDir, d, 'chrome-linux64', 'chrome');
        if (existsSync(candidate)) return candidate;
      }
    }
  }
  return undefined;
}

function buildGridNetwork(side = 6, spacingM = 3000, asymmetric = false) {
  const half = ((side - 1) * spacingM) / 2;
  const stXY = [];
  for (let r = 0; r < side; r++) {
    for (let c = 0; c < side; c++) {
      stXY.push(c * spacingM - half, r * spacingM - half);
    }
  }
  const lineN = [];
  const stops = [];
  for (let r = 0; r < side; r++) {
    lineN.push(side);
    for (let c = 0; c < side; c++) stops.push(r * side + c);
  }
  for (let c = 0; c < side; c++) {
    lineN.push(side);
    for (let r = 0; r < side; r++) stops.push(r * side + c);
  }
  const nStops = stops.length;
  const nLines = lineN.length;
  const times = new Float32Array(nStops * 6);
  const tph = new Float32Array(nLines * 3);
  tph.fill(12);
  const cars = new Float32Array(nLines);
  cars.fill(4);
  let pos = 0;
  for (let l = 0; l < nLines; l++) {
    const len = lineN[l];
    for (let k = 0; k < len; k++) {
      const idx = pos + k;
      const fwd = k + 1 < len ? (asymmetric ? 185 + ((l * 17.37 + k * 29.91) % 113.7) : 240) : 0;
      const bwd = k > 0 ? (asymmetric ? 191 + ((l * 23.19 + k * 31.43) % 107.3) : 240) : 0;
      for (let lev = 0; lev < 3; lev++) {
        times[lev * nStops + idx] = fwd;
        times[(3 + lev) * nStops + idx] = bwd;
      }
    }
    pos += len;
  }
  const hopKm = new Float32Array(nStops * 6);
  pos = 0;
  for (let l = 0; l < nLines; l++) {
    const len = lineN[l];
    for (let k = 0; k < len; k++) {
      const idx = pos + k;
      const fwd = k + 1 < len ? spacingM / 1000 : 0;
      const bwd = k > 0 ? spacingM / 1000 : 0;
      for (let lev = 0; lev < 3; lev++) {
        hopKm[lev * nStops + idx] = fwd;
        hopKm[(3 + lev) * nStops + idx] = bwd;
      }
    }
    pos += len;
  }
  return {
    frame: { kind: 'equirect', lon: -73.985, lat: 40.758 },
    stXY: Array.from(stXY),
    lineN: Array.from(lineN),
    stops: Array.from(stops),
    times: Array.from(times),
    tph: Array.from(tph),
    cars: Array.from(cars),
    fare: { base: 1.0, perKm: 0.1 },
    hopKm: Array.from(hopKm),
  };
}

const packDir = existsSync(join(HERE, '../data/packs/nyc.json'))
  ? join(HERE, '../data/packs')
  : join(HERE, '../data-test/packs');
const nycHeaderPath = join(packDir, 'nyc.json');
const nycBinPath = join(packDir, 'nyc.bin');
assert(existsSync(nycHeaderPath) && existsSync(nycBinPath), `Missing nyc pack in ${packDir}`);

const tmpBuildDir = mkdtempSync(join(tmpdir(), 'tw-webgpu-check-'));
const { rolldown } = await import(
  pathToFileURL(join(HERE, '../app/node_modules/rolldown/dist/index.mjs')).href
);

const gpuBundle = await rolldown({
  input: join(HERE, '../app/src/workers/webgpu/DemandGpuContext.ts'),
  platform: 'browser',
});
const gpuOut = await gpuBundle.generate({ format: 'esm', codeSplitting: false });
const gpuJsCode = gpuOut.output.find((o) => o.type === 'chunk').code;
await gpuBundle.close();

const workerBundle = await rolldown({
  input: join(HERE, '../app/src/workers/demand.worker.ts'),
  platform: 'browser',
  transform: {
    define: {
      'import.meta.env.DEV': 'false',
      'import.meta.glob': '(() => ({}))',
    },
  },
});
const workerDir = join(tmpBuildDir, 'worker');
mkdirSync(workerDir, { recursive: true });
await workerBundle.write({
  file: join(workerDir, 'worker.js'),
  format: 'esm',
  codeSplitting: false,
});
await workerBundle.close();

const simJs = readFileSync(join(HERE, '../app/src/wasm/demand/sim.js'));
const simWasm = readFileSync(join(HERE, '../app/src/wasm/demand/sim_bg.wasm'));
const workerJs = readFileSync(join(workerDir, 'worker.js'));
const workerNoGpuJs = Buffer.concat([
  Buffer.from('try { Object.defineProperty(globalThis, "navigator", { value: {}, configurable: true }); } catch {}\n'),
  workerJs,
]);
const workerShimJs = Buffer.concat([
  Buffer.from(`
if (typeof GPUBuffer !== 'undefined') {
  const origMap = GPUBuffer.prototype.mapAsync;
  let injected = false;
  GPUBuffer.prototype.mapAsync = function (mode, offset, size) {
    if (!injected && this.label === 'staging_od') {
      injected = true;
      self.postMessage({ kind: '__audit_map_entered' });
      return new Promise((res, rej) => {
        setTimeout(() => {
          self.postMessage({ kind: '__audit_map_exited' });
          origMap.call(this, mode, offset, size).then(res, rej);
        }, 120);
      });
    }
    return origMap.call(this, mode, offset, size);
  };
}
`),
  workerJs,
]);
const nycJson = readFileSync(nycHeaderPath);
const nycBin = readFileSync(nycBinPath);
const testNetJson = JSON.stringify(buildGridNetwork(6, 3000, false));
const irrNetJson = JSON.stringify(buildGridNetwork(6, 3000, true));

const server = http.createServer((req, res) => {
  const url = (req.url || '/').split('?')[0];
  if (url === '/') {
    res.writeHead(200, { 'content-type': 'text/html' });
    res.end('<!doctype html><html><body>webgpu-mode-check</body></html>');
    return;
  }
  const routes = {
    '/gpu.js': [gpuJsCode, 'application/javascript'],
    '/worker.js': [workerJs, 'application/javascript'],
    '/worker-nogpu.js': [workerNoGpuJs, 'application/javascript'],
    '/worker-shim.js': [workerShimJs, 'application/javascript'],
    '/sim.js': [simJs, 'application/javascript'],
    '/sim_bg.wasm': [simWasm, 'application/wasm'],
    '/nyc.json': [nycJson, 'application/json'],
    '/nyc.bin': [nycBin, 'application/octet-stream'],
    '/net.json': [testNetJson, 'application/json'],
    '/irr-net.json': [irrNetJson, 'application/json'],
  };
  if (routes[url]) {
    const [body, type] = routes[url];
    res.writeHead(200, { 'content-type': type });
    res.end(body);
    return;
  }
  if (url === '/favicon.ico') {
    res.writeHead(204);
    res.end();
    return;
  }
  console.error(`[check 404] ${url}`);
  res.writeHead(404);
  res.end('not found');
});

let browser;
try {
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const { port } = server.address();
  const pw = await loadPlaywright();
  const executablePath = findChromeExecutable();
  const args =
    process.platform === 'linux'
      ? [
          '--headless=new',
          '--enable-unsafe-webgpu',
          '--enable-features=Vulkan',
          '--ignore-gpu-blocklist',
          '--use-angle=vulkan',
          '--no-sandbox',
        ]
      : ['--headless=new', '--enable-unsafe-webgpu', '--ignore-gpu-blocklist'];

  browser = await pw.chromium.launch({
    ...(executablePath ? { executablePath } : {}),
    headless: true,
    args,
  });

  const page = await browser.newPage();
  const pageErrors = [];
  page.on('pageerror', (err) => pageErrors.push(String(err)));
  page.on('console', (msg) => {
    if (msg.type() === 'error') pageErrors.push(msg.text());
  });
  await page.goto(`http://127.0.0.1:${port}/`);

  const report = await page.evaluate(async () => {
    const mod = await import('/sim.js');
    const wasm = await mod.default('/sim_bg.wasm');
    const { DemandApi } = mod;
    const { DemandGpuContext } = await import('/gpu.js');

    const gpu = await DemandGpuContext.create();
    if (!gpu) {
      return { ok: false, error: 'WebGPU adapter/device unavailable in headless Chromium' };
    }

    function maxRel(a, b, threshold) {
      let maxR = 0;
      let maxIdx = -1;
      for (let i = 0; i < Math.min(a.length, b.length); i++) {
        if (Math.abs(a[i]) > threshold || Math.abs(b[i]) > threshold) {
          const denom = Math.max(Math.abs(a[i]), Math.abs(b[i]), 1e-30);
          const rel = Math.abs(a[i] - b[i]) / denom;
          if (rel > maxR) {
            maxR = rel;
            maxIdx = i;
          }
        }
      }
      return { maxR, maxIdx };
    }

    function l1Rel(a, b) {
      let diff = 0;
      let tot = 0;
      for (let i = 0; i < Math.min(a.length, b.length); i++) {
        diff += Math.abs(a[i] - b[i]);
        tot += Math.abs(a[i]);
      }
      return tot > 0 ? diff / tot : diff;
    }

    async function runGpuPass(api, apiKey, q, crowdTotal) {
      const prep =
        crowdTotal === null ? api.gpu_prepare_solve(q) : api.gpu_prepare_crowd(q, crowdTotal);
      if (prep === 2) return { prep, ok: true, cpuBypassed: true, ms: 0 };
      if (prep !== 1) return { prep, ok: false };
      const t0 = performance.now();
      const res = await gpu.runModeChoicePass({
        apiKey,
        descriptors: api.gpu_stage_descriptors(),
        netMeta: api.gpu_net_meta(),
        passUniforms: api.gpu_pass_uniforms(),
        wasmMemory: wasm.memory,
      });
      if (!res.ok || !res.outputs) {
        api.gpu_abort_pass();
        return { prep, ok: false, reason: res.reason, ms: performance.now() - t0 };
      }
      const o = res.outputs;
      const committed = api.gpu_commit_pass(
        o.od,
        o.accHome,
        o.accQFlat,
        o.railBySub,
        o.walkLostBySub,
        o.unrLostBySub,
        o.uSense,
        o.lostQ,
        o.unrQ,
        o.railBand,
        o.lenFlat,
      );
      return { prep, ok: committed, ms: performance.now() - t0, outputs: o };
    }

    function comparePass(cpu, gpuApi, q, checkRailBand = false, prevGpuRail = null) {
      const sc = Array.from(cpu.summary(q));
      const sg = Array.from(gpuApi.summary(q));
      const summaryRel = {};
      const names = [
        'round',
        'trips',
        'rail',
        'walk',
        'ms',
        'pairs',
        'segs_over_seats',
        'max_load_of_crush',
        'p50',
        'p90',
        'd',
        'd_km',
        'km',
        'unr',
      ];
      for (let i = 0; i < Math.min(sc.length, sg.length); i++) {
        if (Math.abs(sc[i]) <= 1e-9 && Math.abs(sg[i]) <= 1e-9) {
          summaryRel[names[i] ?? `f${i}`] = 0;
        } else {
          const denom = Math.max(Math.abs(sc[i]), Math.abs(sg[i]), 1e-30);
          summaryRel[names[i] ?? `f${i}`] = Math.abs(sc[i] - sg[i]) / denom;
        }
      }
      const subCpu = cpu.sub_sums();
      const subGpu = gpuApi.sub_sums();
      const cmCpu = cpu.cell_modes(subCpu);
      const cmGpu = gpuApi.cell_modes(subGpu);
      const boardCpu = cpu.board(q);
      const boardGpu = gpuApi.board(q);
      const alightCpu = cpu.alight(q);
      const alightGpu = gpuApi.alight(q);
      const segCpu = cpu.seg(q);
      const segGpu = gpuApi.seg(q);
      const lenCpu = cpu.trip_lengths_day(0, 1 << q);
      const lenGpu = gpuApi.trip_lengths_day(0, 1 << q);
      const stCpu = cpu.station_riders_day(0, 1 << q);
      const stGpu = gpuApi.station_riders_day(0, 1 << q);
      let railBandSumRel = 0;
      if (checkRailBand) {
        const rb = Array.from(gpuApi.last_gpu_rail_band());
        const rbSum = rb.reduce((a, b) => a + b, 0);
        const r = sg[0] ?? 0;
        const railGpu = prevGpuRail === null ? (sg[2] ?? 0) : (r + 1) * (sg[2] ?? 0) - r * prevGpuRail;
        railBandSumRel =
          Math.abs(rbSum) <= 1e-9 && Math.abs(railGpu) <= 1e-9
            ? 0
            : Math.abs(rbSum - railGpu) / Math.max(Math.abs(rbSum), Math.abs(railGpu), 1e-30);
      }
      return {
        summaryRel,
        subSumsMaxRel: maxRel(subCpu, subGpu, 1.0).maxR,
        cellModesMaxRel: maxRel(cmCpu, cmGpu, 1.0).maxR,
        boardMaxRel: maxRel(boardCpu, boardGpu, 1.0).maxR,
        alightMaxRel: maxRel(alightCpu, alightGpu, 1.0).maxR,
        segMaxRel: maxRel(segCpu, segGpu, 1.0).maxR,
        segL1Rel: l1Rel(segCpu, segGpu),
        lenMaxRel: maxRel(lenCpu, lenGpu, 1.0).maxR,
        stRiderMaxRel: maxRel(stCpu, stGpu, 1.0).maxR,
        railBandSumRel,
      };
    }

    const [net, irrNet, nycHdr, nycBuf] = await Promise.all([
      fetch('/net.json').then((r) => r.json()),
      fetch('/irr-net.json').then((r) => r.json()),
      fetch('/nyc.json').then((r) => r.text()),
      fetch('/nyc.bin').then((r) => r.arrayBuffer()),
    ]);
    const nycBytes = new Uint8Array(nycBuf);
    const flatXy = new Float32Array(net.stXY);
    const lineN = new Uint32Array(net.lineN);
    const stops = new Uint32Array(net.stops);
    const times = new Float32Array(net.times);
    const tph = new Float32Array(net.tph);
    const cars = new Float32Array(net.cars);

    // 1. Parity across solve(0), crowd round 1, and crowd round 2 on nyc.
    const cpu = new DemandApi(nycHdr, nycBytes);
    const gpuApi = new DemandApi(nycHdr, nycBytes);
    cpu.set_network(flatXy, lineN, stops, times, tph, cars);
    gpuApi.set_network(flatXy, lineN, stops, times, tph, cars);

    const tc0 = performance.now();
    cpu.solve(0);
    const cpuSolveMs = performance.now() - tc0;
    const g0 = await runGpuPass(gpuApi, 1, 0, null);
    if (!g0.ok || g0.prep !== 1) return { ok: false, error: `GPU solve(0) failed: ${g0.reason}` };
    const prevRail0 = gpuApi.summary(0)[2];
    const paritySolve = comparePass(cpu, gpuApi, 0, true);

    const concatTotal = (api, q) => new Float32Array([...api.seg(q), ...api.board(q), ...api.alight(q)]);

    const cpuTot1 = concatTotal(cpu, 0);
    const gpuTot1 = concatTotal(gpuApi, 0);
    if (!cpu.crowd_with_external(0, cpuTot1)) return { ok: false, error: 'CPU crowd r1 failed' };
    const g1 = await runGpuPass(gpuApi, 1, 0, gpuTot1);
    if (!g1.ok) return { ok: false, error: `GPU crowd r1 failed: ${g1.reason} (prep=${g1.prep})` };
    const prevRail1 = gpuApi.summary(0)[2];
    const parityCrowd1 = comparePass(cpu, gpuApi, 0, true, prevRail0);

    const cpuTot2 = concatTotal(cpu, 0);
    const gpuTot2 = concatTotal(gpuApi, 0);
    if (!cpu.crowd_with_external(0, cpuTot2)) return { ok: false, error: 'CPU crowd r2 failed' };
    const tc2 = performance.now();
    const g2 = await runGpuPass(gpuApi, 1, 0, gpuTot2);
    const gpuWarmCrowdMs = performance.now() - tc2;
    if (!g2.ok) return { ok: false, error: `GPU crowd r2 failed: ${g2.reason}` };
    const parityCrowd2 = comparePass(cpu, gpuApi, 0, true, prevRail1);

    // 1b. All 5 periods (q=0..4) parity including station_riders_day and last_gpu_rail_band().
    const parityPeriods = [];
    for (let q = 0; q < 5; q++) {
      cpu.solve(q);
      const gq = await runGpuPass(gpuApi, 1, q, null);
      if (!gq.ok || gq.prep !== 1) return { ok: false, error: `GPU solve(${q}) failed: ${gq.reason}` };
      parityPeriods.push({ q, ...comparePass(cpu, gpuApi, q, true) });
    }

    // 1c. Asymmetric closed-loop 3-round crowding parity.
    const irrCpu = new DemandApi(nycHdr, nycBytes);
    const irrGpu = new DemandApi(nycHdr, nycBytes);
    for (const a of [irrCpu, irrGpu]) {
      a.set_network(
        new Float32Array(irrNet.stXY),
        new Uint32Array(irrNet.lineN),
        new Uint32Array(irrNet.stops),
        new Float32Array(irrNet.times),
        new Float32Array(irrNet.tph),
        new Float32Array(irrNet.cars),
      );
    }
    irrCpu.solve(0);
    const gIrr0 = await runGpuPass(irrGpu, 10, 0, null);
    if (!gIrr0.ok) return { ok: false, error: `Asymmetric closed-loop solve failed: ${gIrr0.reason}` };
    let irrPrevRail = irrGpu.summary(0)[2];
    const irrRounds = [];
    for (let r = 1; r <= 3; r++) {
      if (!irrCpu.crowd_with_external(0, concatTotal(irrCpu, 0))) {
        return { ok: false, error: `Asymmetric CPU crowd r=${r} failed` };
      }
      const gr = await runGpuPass(irrGpu, 10, 0, concatTotal(irrGpu, 0));
      if (!gr.ok) return { ok: false, error: `Asymmetric GPU crowd r=${r} failed: ${gr.reason}` };
      irrRounds.push({ r, ...comparePass(irrCpu, irrGpu, 0, true, irrPrevRail) });
      irrPrevRail = irrGpu.summary(0)[2];
    }
    irrCpu.free();
    irrGpu.free();

    // 1d. Edge cases: s=0 empty network & disconnected lines.
    const emptyCpu = new DemandApi(nycHdr, nycBytes);
    const emptyGpu = new DemandApi(nycHdr, nycBytes);
    for (const a of [emptyCpu, emptyGpu]) {
      a.set_network(
        new Float32Array(0),
        new Uint32Array(0),
        new Uint32Array(0),
        new Float32Array(0),
        new Float32Array(0),
        new Float32Array(0),
      );
    }
    emptyCpu.solve(0);
    const gEmpty = await runGpuPass(emptyGpu, 11, 0, null);
    const emptyParity = gEmpty.ok ? comparePass(emptyCpu, emptyGpu, 0, false) : null;
    emptyCpu.free();
    emptyGpu.free();

    const discCpu = new DemandApi(nycHdr, nycBytes);
    const discGpu = new DemandApi(nycHdr, nycBytes);
    const discTimes = new Float32Array(6 * 6);
    for (let lev = 0; lev < 3; lev++) {
      // fwd: stops 0->1->2 (line 0) and 3->4->5 (line 1)
      discTimes[lev * 6 + 0] = 180;
      discTimes[lev * 6 + 1] = 180;
      discTimes[lev * 6 + 3] = 180;
      discTimes[lev * 6 + 4] = 180;
      // bwd: stops 2->1->0 (line 0) and 5->4->3 (line 1)
      discTimes[(3 + lev) * 6 + 1] = 180;
      discTimes[(3 + lev) * 6 + 2] = 180;
      discTimes[(3 + lev) * 6 + 4] = 180;
      discTimes[(3 + lev) * 6 + 5] = 180;
    }
    const discArgs = [
      new Float32Array([-3000, 0, -2000, 0, -1000, 0, 2000, 3000, 3000, 3000, 4000, 3000]),
      new Uint32Array([3, 3]),
      new Uint32Array([0, 1, 2, 3, 4, 5]),
      discTimes,
      new Float32Array([12, 12, 12, 8, 8, 8]),
      new Float32Array([4, 3]),
    ];
    for (const a of [discCpu, discGpu]) a.set_network(...discArgs);
    discCpu.solve(0);
    const gDisc = await runGpuPass(discGpu, 12, 0, null);
    const discParity = gDisc.ok ? comparePass(discCpu, discGpu, 0, false) : null;
    discCpu.free();
    discGpu.free();

    // 1e. 2D workgroup chunking for s >= 2048 (`stationExp` = 65536 > 65535 workgroups).
    const sBig = 2048;
    const stXyBig = new Float32Array(sBig * 2);
    const stopsBig = new Uint32Array(sBig);
    for (let i = 0; i < sBig; i++) {
      stXyBig[2 * i] = (i % 64) * 500 - 15000;
      stXyBig[2 * i + 1] = Math.floor(i / 64) * 500 - 8000;
      stopsBig[i] = i;
    }
    const bigApi = new DemandApi(nycHdr, nycBytes);
    bigApi.set_network(
      stXyBig,
      new Uint32Array([sBig]),
      stopsBig,
      new Float32Array(sBig * 6).fill(60),
      new Float32Array([12, 12, 12]),
      new Float32Array([4]),
    );
    const bigMeta = bigApi.gpu_prepare_solve(0) === 1 ? bigApi.gpu_net_meta() : null;
    let largeSResult = { prep: bigMeta ? 1 : 0, ok: false, metaLen: bigMeta?.length ?? 0 };
    if (bigMeta) {
      const bigRes = await gpu.runModeChoicePass({
        apiKey: 13,
        descriptors: bigApi.gpu_stage_descriptors(),
        netMeta: bigMeta,
        passUniforms: bigApi.gpu_pass_uniforms(),
        wasmMemory: wasm.memory,
      });
      bigApi.gpu_abort_pass();
      largeSResult = { prep: 1, ok: bigRes.ok && gpu.isHealthy, metaLen: bigMeta.length };
    }
    bigApi.free();

    // 1f. Mid-solve mapAsync rejection: zero leaked error scopes / unmapped staging buffers, next GPU pass works.
    const rejApi = new DemandApi(nycHdr, nycBytes);
    rejApi.set_network(flatXy, lineN, stops, times, tph, cars);
    const rejPrep = rejApi.gpu_prepare_solve(0);
    let origMapProto = GPUBuffer.prototype.mapAsync;
    let injectFail = true;
    GPUBuffer.prototype.mapAsync = function (mode, offset, size) {
      if (injectFail && this.label === 'staging_od') {
        injectFail = false;
        return Promise.reject(new Error('Simulated mapAsync rejection'));
      }
      return origMapProto.call(this, mode, offset, size);
    };
    const rejRes = await gpu.runModeChoicePass({
      apiKey: 1,
      descriptors: rejApi.gpu_stage_descriptors(),
      netMeta: rejApi.gpu_net_meta(),
      passUniforms: rejApi.gpu_pass_uniforms(),
      wasmMemory: wasm.memory,
    });
    GPUBuffer.prototype.mapAsync = origMapProto;
    rejApi.gpu_abort_pass();
    rejApi.solve(0);
    const cpuRejRef = new DemandApi(nycHdr, nycBytes);
    cpuRejRef.set_network(flatXy, lineN, stops, times, tph, cars);
    cpuRejRef.solve(0);
    const rejCmp = comparePass(cpuRejRef, rejApi, 0, false);
    cpuRejRef.free();
    rejApi.free();
    const nextAfterRej = await runGpuPass(gpuApi, 1, 0, null);
    const mapRejectRecovered =
      rejPrep === 1 &&
      !rejRes.ok &&
      gpu.isHealthy &&
      nextAfterRej.ok &&
      gpu.isHealthy &&
      rejCmp.subSumsMaxRel === 0 &&
      rejCmp.segL1Rel === 0;

    // 2. Same-GPU bitwise determinism across independent passes.
    const apiDet = new DemandApi(nycHdr, nycBytes);
    apiDet.set_network(flatXy, lineN, stops, times, tph, cars);
    const gDet = await runGpuPass(apiDet, 2, 0, null);
    let ulpMismatches = 0;
    for (const [a, b] of [
      [g0.outputs.od, gDet.outputs.od],
      [g0.outputs.accHome, gDet.outputs.accHome],
      [g0.outputs.railBand, gDet.outputs.railBand],
      [g0.outputs.lenFlat, gDet.outputs.lenFlat],
    ]) {
      const ua = new Uint32Array(a.buffer, a.byteOffset, a.length);
      const ub = new Uint32Array(b.buffer, b.byteOffset, b.length);
      for (let i = 0; i < ua.length; i++) {
        if (ua[i] !== ub[i]) ulpMismatches++;
      }
    }
    apiDet.free();

    // 3. Simulated mid-solve device loss -> gpu_abort_pass() -> clean CPU fallback.
    const apiLost = new DemandApi(nycHdr, nycBytes);
    apiLost.set_network(flatXy, lineN, stops, times, tph, cars);
    const prepLost = apiLost.gpu_prepare_solve(0);
    gpu.device.destroy();
    const lostRes = await gpu.runModeChoicePass({
      apiKey: 3,
      descriptors: apiLost.gpu_stage_descriptors(),
      netMeta: apiLost.gpu_net_meta(),
      passUniforms: apiLost.gpu_pass_uniforms(),
      wasmMemory: wasm.memory,
    });
    apiLost.gpu_abort_pass();
    apiLost.solve(0);

    const cpuFresh = new DemandApi(nycHdr, nycBytes);
    cpuFresh.set_network(flatXy, lineN, stops, times, tph, cars);
    cpuFresh.solve(0);
    const lostFallbackCmp = comparePass(cpuFresh, apiLost, 0);
    const deviceLostRecovered = !lostRes.ok && !gpu.isHealthy && lostFallbackCmp.subSumsMaxRel === 0 && lostFallbackCmp.segL1Rel === 0;

    cpu.free();
    gpuApi.free();
    apiLost.free();
    cpuFresh.free();

    // 4. End-to-end DedicatedWorker checks: WebGPU enabled vs noGpu=true (?perfOff=webgpu) vs missing navigator.gpu + async re-entrancy.
    async function runDedicatedWorker(workerUrl, noGpu) {
      return await new Promise((resolve, reject) => {
        const w = new Worker(workerUrl, { type: 'module' });
        const summaries = {};
        const sums = {};
        let resolved = false;
        w.onerror = (e) => {
          if (!resolved) {
            resolved = true;
            w.terminate();
            reject(new Error(`Worker error: ${e.message}`));
          }
        };
        w.onmessage = ({ data: m }) => {
          if (m.kind === 'failed' || m.kind === 'dead') {
            if (!resolved) {
              resolved = true;
              w.terminate();
              reject(new Error(`Worker ${m.kind}: ${m.error}`));
            }
            return;
          }
          if (m.kind === 'ready') {
            w.postMessage({
              kind: 'solve',
              version: 1,
              net: {
                frame: net.frame,
                stXY: new Float32Array(net.stXY),
                lineN: new Uint32Array(net.lineN),
                stops: new Uint32Array(net.stops),
                times: new Float32Array(net.times),
                tph: new Float32Array(net.tph),
                cars: new Float32Array(net.cars),
                fare: net.fare,
                hopKm: new Float32Array(net.hopKm),
              },
              rounds: 2,
            });
          } else if (m.kind === 'result' && !m.cluster) {
            summaries[m.period] = Array.from(m.citySummary?.nyc ?? m.summary);
          } else if (m.kind === 'sums') {
            sums[m.round] = m.sums;
          } else if (m.kind === 'done') {
            resolved = true;
            w.terminate();
            resolve({ summaries, sums });
          }
        };
        const binCopy = nycBuf.slice(0);
        w.postMessage(
          { kind: 'init', packs: [{ id: 'nyc', header: nycHdr, bin: binCopy }], periods: [0, 1], noGpu },
          [binCopy],
        );
      });
    }

    async function runWorkerReentrancyCheck() {
      return await new Promise((resolve, reject) => {
        const w = new Worker('/worker-shim.js', { type: 'module' });
        let insideMap = false;
        let interleavedCount = 0;
        let done = false;
        w.onerror = (e) => { w.terminate(); reject(new Error(`Worker shim error: ${e.message}`)); };
        w.onmessage = ({ data: m }) => {
          if (m.kind === '__audit_map_entered') {
            insideMap = true;
            w.postMessage({ kind: 'subSums', id: 901, city: 'nyc', version: 1, round: 0, wait: false });
            w.postMessage({ kind: 'station', id: 902, city: 'nyc', version: 1, round: 0, station: 0, wait: false });
            w.postMessage({ kind: 'close', ids: ['nonexistent'] });
            w.postMessage({ kind: 'open', packs: [] });
          } else if (m.kind === '__audit_map_exited') {
            insideMap = false;
          } else if (['subSums', 'station', 'opened'].includes(m.kind)) {
            if (insideMap) interleavedCount++;
          } else if (m.kind === 'dead' || m.kind === 'failed') {
            w.terminate();
            reject(new Error(`Worker shim ${m.kind}: ${m.error}`));
          } else if (m.kind === 'ready') {
            w.postMessage({
              kind: 'solve',
              version: 1,
              net: {
                frame: net.frame,
                stXY: new Float32Array(net.stXY),
                lineN: new Uint32Array(net.lineN),
                stops: new Uint32Array(net.stops),
                times: new Float32Array(net.times),
                tph: new Float32Array(net.tph),
                cars: new Float32Array(net.cars),
                fare: net.fare,
                hopKm: new Float32Array(net.hopKm),
              },
              rounds: 1,
            });
          } else if (m.kind === 'done') {
            done = true;
            w.terminate();
            resolve({ done, interleavedCount });
          }
        };
        const binCopy = nycBuf.slice(0);
        w.postMessage(
          { kind: 'init', packs: [{ id: 'nyc', header: nycHdr, bin: binCopy }], periods: [0], noGpu: false },
          [binCopy],
        );
      });
    }

    const [workerGpu, workerCpu, workerNoGpu, workerReentrancy] = await Promise.all([
      runDedicatedWorker('/worker.js', false),
      runDedicatedWorker('/worker.js', true),
      runDedicatedWorker('/worker-nogpu.js', false),
      runWorkerReentrancyCheck(),
    ]);
    const workerSummaryRel = {};
    const workerNoGpuSummaryRel = {};
    for (const q of [0, 1]) {
      const a = workerGpu.summaries[q];
      const b = workerCpu.summaries[q];
      const c = workerNoGpu.summaries[q];
      workerSummaryRel[q] = a.map((v, i) =>
        Math.abs(v) <= 1e-9 && Math.abs(b[i]) <= 1e-9
          ? 0
          : Math.abs(v - b[i]) / Math.max(Math.abs(v), Math.abs(b[i]), 1e-30),
      );
      workerNoGpuSummaryRel[q] = c.map((v, i) =>
        Math.abs(v) <= 1e-9 && Math.abs(b[i]) <= 1e-9
          ? 0
          : Math.abs(v - b[i]) / Math.max(Math.abs(v), Math.abs(b[i]), 1e-30),
      );
    }
    const workerSumsMaxRel = maxRel(workerGpu.sums[2], workerCpu.sums[2], 1.0).maxR;
    const workerNoGpuSumsMaxRel = maxRel(workerNoGpu.sums[2], workerCpu.sums[2], 1.0).maxR;

    return {
      ok: true,
      cpuSolveMs,
      gpuColdSolveMs: g0.ms,
      gpuWarmCrowdMs,
      paritySolve,
      parityCrowd1,
      parityCrowd2,
      parityPeriods,
      irrRounds,
      gEmptyPrep: gEmpty.prep,
      emptyParity,
      discPrep: gDisc.prep,
      discParity,
      largeSResult,
      mapRejectRecovered,
      ulpMismatches,
      prepLost,
      deviceLostRecovered,
      workerSummaryRel,
      workerNoGpuSummaryRel,
      workerSumsMaxRel,
      workerNoGpuSumsMaxRel,
      workerReentrancy,
    };
  });

  assert.equal(pageErrors.length, 0, `Browser errors: ${pageErrors.join('; ')}`);
  assert(report.ok, report.error);
  assert.equal(report.ulpMismatches, 0, `Non-deterministic ULPs: ${report.ulpMismatches}`);
  assert.equal(report.prepLost, 1);
  assert(report.deviceLostRecovered, 'Simulated device.lost did not recover cleanly to CPU fallback');
  assert(report.mapRejectRecovered, 'Mid-solve mapAsync rejection leaked scope/state or broke recovery');
  assert(report.largeSResult.ok && report.largeSResult.metaLen === 38, `s=2048 2D workgroup dispatch failed: ${JSON.stringify(report.largeSResult)}`);
  assert.equal(report.gEmptyPrep, 2, `Empty network s=0 should bypass GPU with prep=2`);
  assert(report.emptyParity && report.emptyParity.subSumsMaxRel === 0, 'Empty network s=0 parity failed');
  assert(report.discParity && report.discParity.subSumsMaxRel <= 1e-5 && report.discParity.segL1Rel <= 1e-6, 'Disconnected lines parity failed');
  assert(report.workerReentrancy.done && report.workerReentrancy.interleavedCount === 0, `Worker re-entrancy interleaved=${report.workerReentrancy.interleavedCount}`);

  const assertParity = (label, cmp) => {
    for (const k of ['round', 'trips', 'rail', 'walk', 'pairs', 'segs_over_seats', 'max_load_of_crush', 'p50', 'p90', 'd', 'd_km', 'km', 'unr']) {
      assert(cmp.summaryRel[k] <= 1e-5, `${label} summary.${k} rel=${cmp.summaryRel[k]}`);
    }
    assert(cmp.subSumsMaxRel <= 1e-5, `${label} subSumsMaxRel=${cmp.subSumsMaxRel}`);
    assert(cmp.cellModesMaxRel <= 1e-5, `${label} cellModesMaxRel=${cmp.cellModesMaxRel}`);
    assert(cmp.boardMaxRel <= 1e-5, `${label} boardMaxRel=${cmp.boardMaxRel}`);
    assert(cmp.alightMaxRel <= 1e-5, `${label} alightMaxRel=${cmp.alightMaxRel}`);
    assert(cmp.segMaxRel <= 1e-5, `${label} segMaxRel=${cmp.segMaxRel}`);
    assert(cmp.segL1Rel <= 1e-6, `${label} segL1Rel=${cmp.segL1Rel}`);
    assert(cmp.lenMaxRel <= 5e-5, `${label} lenMaxRel=${cmp.lenMaxRel}`);
    assert(cmp.stRiderMaxRel <= 1e-5, `${label} stRiderMaxRel=${cmp.stRiderMaxRel}`);
    assert(cmp.railBandSumRel <= 1e-5, `${label} railBandSumRel=${cmp.railBandSumRel}`);
  };

  for (const [label, cmp] of [
    ['solve(0)', report.paritySolve],
    ['crowd r1', report.parityCrowd1],
    ['crowd r2', report.parityCrowd2],
  ]) {
    assertParity(label, cmp);
  }
  for (const p of report.parityPeriods) {
    assertParity(`solve(q=${p.q})`, p);
  }
  for (const r of report.irrRounds) {
    assertParity(`irr crowd r=${r.r}`, r);
  }

  for (const q of [0, 1]) {
    assert.equal(report.workerSummaryRel[q].length, 14, `worker q=${q} unexpected summary length`);
    assert.equal(report.workerNoGpuSummaryRel[q].length, 14, `workerNoGpu q=${q} unexpected summary length`);
    for (let i = 0; i < report.workerSummaryRel[q].length; i++) {
      if (i === 4) continue; // index 4 is wall-clock st.ms
      assert(
        report.workerSummaryRel[q][i] <= 1e-5,
        `worker q=${q} summary[${i}] rel=${report.workerSummaryRel[q][i]}`,
      );
      assert(
        report.workerNoGpuSummaryRel[q][i] <= 1e-5,
        `workerNoGpu q=${q} summary[${i}] rel=${report.workerNoGpuSummaryRel[q][i]}`,
      );
    }
  }
  assert(report.workerSumsMaxRel <= 1e-5, `workerSumsMaxRel=${report.workerSumsMaxRel}`);
  assert(report.workerNoGpuSumsMaxRel <= 1e-5, `workerNoGpuSumsMaxRel=${report.workerNoGpuSumsMaxRel}`);

  console.log(JSON.stringify(report, null, 2));
} finally {
  if (browser) await browser.close().catch(() => {});
  server.close();
  rmSync(tmpBuildDir, { recursive: true, force: true });
}
