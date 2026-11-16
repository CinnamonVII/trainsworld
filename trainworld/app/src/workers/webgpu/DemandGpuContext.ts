import {
  ACC_HOME_WGSL,
  BUILD_AB_ROW_WGSL,
  BUILD_AB_Y_WGSL,
  computeBufferLayout,
  GPU_U_BATCH,
  KE,
  LEN_PHASE1_WGSL,
  LEN_PHASE2_WGSL,
  MODE_PAIRS_WGSL,
  N_BANDS,
  N_LEN,
  REDUCE_BANDS_WGSL,
  REDUCE_OZ_LEN_WGSL,
  REDUCE_Q_WGSL,
  REDUCE_U_WGSL,
  SLOW_APPLY_OD_WGSL,
  SLOW_APPLY_REST_WGSL,
  SLOW_BIN_WGSL,
  SPLIT_TO_OD_WGSL,
  STATION_EXP_WGSL,
  UQ_ZERO_WGSL,
  U_SETUP_WGSL,
  W_LEN,
  ZERO_PASS_WGSL,
} from "./shaders";
import { requestWebGpuAdapter, requestWebGpuDevice } from "./adapter";

export interface GpuPassStageInputs {
  apiKey: number;
  descriptors: Uint32Array;
  netMeta: Uint32Array;
  passUniforms: Float32Array;
  wasmMemory: WebAssembly.Memory;
}

export interface GpuPassOutputs {
  od: Float32Array;
  accHome: Float32Array;
  accQFlat: Float32Array;
  railBySub: Float32Array;
  walkLostBySub: Float32Array;
  unrLostBySub: Float32Array;
  uSense: Float32Array;
  lostQ: Float32Array;
  unrQ: Float32Array;
  railBand: Float32Array;
  lenFlat: Float32Array;
}

export interface GpuPassResult {
  ok: boolean;
  reason?: string;
  outputs?: GpuPassOutputs;
  kernelMs?: Record<string, number>;
}

interface ResidentPeriodBuffers {
  key: string;
  nVisits: number;
  offOuPeriod: number;
  offOzVisitOff: number;
  batchVRange: Uint32Array;
  maxVBatch: number;
  perF32Buf: GPUBuffer;
  perU32Buf: GPUBuffer;
  approxBytes: number;
}

interface ResidentNetworkBuffers {
  key: string;
  meta: Uint32Array;
  layout: ReturnType<typeof computeBufferLayout>;
  batchActiveALoHi: Uint32Array;
  netF32Buf: GPUBuffer;
  netU32Buf: GPUBuffer;
  erBuf: GPUBuffer;
  odBuf: GPUBuffer;
  outScalarsBuf: GPUBuffer;
  skimF32Buf: GPUBuffer;
  skimU32Buf: GPUBuffer;
  batchF32Buf: GPUBuffer;
  batchU32Buf: GPUBuffer;
  uniformBuf: GPUBuffer;
  stagingOdBuf: GPUBuffer;
  stagingScalarsBuf: GPUBuffer;
  maxVBatchAlloc: number;
  uniformSlotsAlloc: number;
  approxBytes: number;
  lastUsedTick: number;
}

interface Pipelines {
  bglDyn0: GPUBindGroupLayout;
  stationExp: GPUComputePipeline;
  zeroPass: GPUComputePipeline;
  uqZero: GPUComputePipeline;
  uSetup: GPUComputePipeline;
  buildAbRow: GPUComputePipeline;
  modePairs: GPUComputePipeline;
  buildAbY: GPUComputePipeline;
  splitToOd: GPUComputePipeline;
  accHome: GPUComputePipeline;
  reduceU: GPUComputePipeline;
  reduceQ: GPUComputePipeline;
  reduceOzLen: GPUComputePipeline;
  lenPhase1: GPUComputePipeline;
  lenPhase2: GPUComputePipeline;
  slowPrepU: GPUComputePipeline;
  slowScatterU: GPUComputePipeline;
  slowSortU: GPUComputePipeline;
  slowBktCount: GPUComputePipeline;
  slowBktPrefix1: GPUComputePipeline;
  slowBktPrefix2: GPUComputePipeline;
  slowBktScatter: GPUComputePipeline;
  slowApplyOd: GPUComputePipeline;
  slowApplyQ: GPUComputePipeline;
  slowApplyHome: GPUComputePipeline;
  reduceBands: GPUComputePipeline;
}

const MAX_NET_CACHE = 2;
const MAX_PERIOD_CACHE = 8;

function align4(bytes: number): number {
  return (bytes + 3) & ~3;
}

export class DemandGpuContext {
  readonly device: GPUDevice;
  private readonly pipelines: Pipelines;
  private readonly hasTimestamps: boolean;
  private readonly netCache = new Map<string, ResidentNetworkBuffers>();
  private readonly periodCache = new Map<string, ResidentPeriodBuffers>();
  private tick = 0;
  private brokenReason: string | null = null;
  private disposed = false;

  private constructor(device: GPUDevice, pipelines: Pipelines, hasTimestamps: boolean) {
    this.device = device;
    this.pipelines = pipelines;
    this.hasTimestamps = hasTimestamps;

    device.lost.then((info) => {
      this.brokenReason = `WebGPU device lost (${info.reason}): ${info.message}`;
      this.destroyAllCaches();
    });
    device.addEventListener("uncapturederror", (ev) => {
      const msg = (ev as GPUUncapturedErrorEvent).error?.message ?? "uncaptured WebGPU error";
      this.brokenReason = msg;
      this.dispose();
    });
  }

  static async create(opts?: { timestamps?: boolean }): Promise<DemandGpuContext | null> {
    if (typeof navigator === "undefined" || !("gpu" in navigator) || !navigator.gpu) {
      return null;
    }
    try {
      const { adapter } = await requestWebGpuAdapter();
      if (!adapter) {
        return null;
      }
      const wantTs = !!opts?.timestamps && adapter.features.has("timestamp-query");
      const requiredFeatures: GPUFeatureName[] = wantTs ? ["timestamp-query"] : [];
      const device = await requestWebGpuDevice(adapter, {
        maxStorageBindingMiB: 512,
        maxBufferMiB: 1024,
        requiredFeatures,
      });
      const pipelines = await DemandGpuContext.compilePipelinesAsync(device);
      return new DemandGpuContext(device, pipelines, wantTs);
    } catch {
      return null;
    }
  }

  get isHealthy(): boolean {
    return !this.disposed && this.brokenReason === null;
  }

  get failureReason(): string | null {
    return this.brokenReason;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.brokenReason ??= "WebGPU context disposed";
    this.destroyAllCaches();
    try {
      this.device.destroy();
    } catch {
      // ignore if already destroyed
    }
  }

  private static makePipelineAsync(
    device: GPUDevice,
    code: string,
    entryPoint: string,
    storageBindings: Array<{ binding: number; type: GPUBufferBindingType }>,
  ): Promise<GPUComputePipeline> {
    const entries: GPUBindGroupLayoutEntry[] = [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 256 },
      },
      ...storageBindings.map((b) => ({
        binding: b.binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: b.type },
      })),
    ];
    const bgl = device.createBindGroupLayout({ entries });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
    const module = device.createShaderModule({ code });
    return device.createComputePipelineAsync({
      layout,
      compute: { module, entryPoint },
    });
  }

  private static async compilePipelinesAsync(device: GPUDevice): Promise<Pipelines> {
    const bglDyn0 = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 256 },
        },
      ],
    });
    const ro = "read-only-storage" as const;
    const rw = "storage" as const;

    const slowMod = device.createShaderModule({ code: SLOW_BIN_WGSL });
    const slowLayout = device.createPipelineLayout({
      bindGroupLayouts: [
        device.createBindGroupLayout({
          entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 256 } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: rw } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: rw } },
          ],
        }),
      ],
    });

    const slowRestMod = device.createShaderModule({ code: SLOW_APPLY_REST_WGSL });
    const slowRestLayout = device.createPipelineLayout({
      bindGroupLayouts: [
        device.createBindGroupLayout({
          entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 256 } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: ro } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: rw } },
          ],
        }),
      ],
    });

    const [
      stationExp,
      zeroPass,
      uqZero,
      uSetup,
      buildAbRow,
      modePairs,
      buildAbY,
      splitToOd,
      accHome,
      reduceU,
      reduceQ,
      reduceOzLen,
      lenPhase1,
      lenPhase2,
      slowPrepU,
      slowScatterU,
      slowSortU,
      slowBktCount,
      slowBktPrefix1,
      slowBktPrefix2,
      slowBktScatter,
      slowApplyOd,
      slowApplyQ,
      slowApplyHome,
      reduceBands,
    ] = await Promise.all([
      DemandGpuContext.makePipelineAsync(device, STATION_EXP_WGSL, "main", [
        { binding: 1, type: ro },
        { binding: 2, type: ro },
        { binding: 3, type: rw },
      ]),
      DemandGpuContext.makePipelineAsync(device, ZERO_PASS_WGSL, "main", [
        { binding: 1, type: rw },
        { binding: 2, type: rw },
        { binding: 3, type: rw },
      ]),
      DemandGpuContext.makePipelineAsync(device, UQ_ZERO_WGSL, "main", [
        { binding: 1, type: rw },
        { binding: 2, type: rw },
      ]),
      DemandGpuContext.makePipelineAsync(device, U_SETUP_WGSL, "main", [
        { binding: 1, type: ro },
        { binding: 2, type: ro },
        { binding: 3, type: ro },
        { binding: 4, type: ro },
        { binding: 5, type: ro },
        { binding: 6, type: rw },
        { binding: 7, type: rw },
      ]),
      DemandGpuContext.makePipelineAsync(device, BUILD_AB_ROW_WGSL, "main", [
        { binding: 1, type: ro },
        { binding: 2, type: ro },
        { binding: 3, type: ro },
        { binding: 4, type: rw },
        { binding: 5, type: ro },
      ]),
      DemandGpuContext.makePipelineAsync(device, MODE_PAIRS_WGSL, "main", [
        { binding: 1, type: ro },
        { binding: 2, type: ro },
        { binding: 3, type: ro },
        { binding: 4, type: ro },
        { binding: 5, type: ro },
        { binding: 6, type: rw },
        { binding: 7, type: rw },
      ]),
      DemandGpuContext.makePipelineAsync(device, BUILD_AB_Y_WGSL, "main", [
        { binding: 1, type: ro },
        { binding: 2, type: ro },
        { binding: 3, type: rw },
        { binding: 4, type: ro },
      ]),
      DemandGpuContext.makePipelineAsync(device, SPLIT_TO_OD_WGSL, "main", [
        { binding: 1, type: ro },
        { binding: 2, type: ro },
        { binding: 3, type: ro },
        { binding: 4, type: ro },
        { binding: 5, type: ro },
        { binding: 6, type: rw },
      ]),
      DemandGpuContext.makePipelineAsync(device, ACC_HOME_WGSL, "main", [
        { binding: 1, type: ro },
        { binding: 2, type: ro },
        { binding: 3, type: ro },
        { binding: 4, type: ro },
        { binding: 5, type: ro },
        { binding: 6, type: rw },
      ]),
      DemandGpuContext.makePipelineAsync(device, REDUCE_U_WGSL, "main", [
        { binding: 1, type: ro },
        { binding: 2, type: ro },
        { binding: 3, type: rw },
      ]),
      DemandGpuContext.makePipelineAsync(device, REDUCE_Q_WGSL, "main", [
        { binding: 1, type: ro },
        { binding: 2, type: ro },
        { binding: 3, type: ro },
        { binding: 4, type: rw },
      ]),
      DemandGpuContext.makePipelineAsync(device, REDUCE_OZ_LEN_WGSL, "main", [
        { binding: 1, type: ro },
        { binding: 2, type: ro },
        { binding: 3, type: rw },
      ]),
      DemandGpuContext.makePipelineAsync(device, LEN_PHASE1_WGSL, "main", [
        { binding: 1, type: ro },
        { binding: 2, type: ro },
        { binding: 3, type: ro },
        { binding: 4, type: rw },
      ]),
      DemandGpuContext.makePipelineAsync(device, LEN_PHASE2_WGSL, "main", [
        { binding: 1, type: ro },
        { binding: 2, type: rw },
      ]),
      device.createComputePipelineAsync({
        layout: slowLayout,
        compute: { module: slowMod, entryPoint: "slow_prep_u" },
      }),
      device.createComputePipelineAsync({
        layout: slowLayout,
        compute: { module: slowMod, entryPoint: "slow_scatter_u" },
      }),
      device.createComputePipelineAsync({
        layout: slowLayout,
        compute: { module: slowMod, entryPoint: "slow_sort_u" },
      }),
      device.createComputePipelineAsync({
        layout: slowLayout,
        compute: { module: slowMod, entryPoint: "slow_bkt_count" },
      }),
      device.createComputePipelineAsync({
        layout: slowLayout,
        compute: { module: slowMod, entryPoint: "slow_bkt_prefix1" },
      }),
      device.createComputePipelineAsync({
        layout: slowLayout,
        compute: { module: slowMod, entryPoint: "slow_bkt_prefix2" },
      }),
      device.createComputePipelineAsync({
        layout: slowLayout,
        compute: { module: slowMod, entryPoint: "slow_bkt_scatter" },
      }),
      DemandGpuContext.makePipelineAsync(device, SLOW_APPLY_OD_WGSL, "main", [
        { binding: 1, type: ro },
        { binding: 2, type: rw },
      ]),
      device.createComputePipelineAsync({
        layout: slowRestLayout,
        compute: { module: slowRestMod, entryPoint: "slow_apply_q" },
      }),
      device.createComputePipelineAsync({
        layout: slowRestLayout,
        compute: { module: slowRestMod, entryPoint: "slow_apply_home" },
      }),
      DemandGpuContext.makePipelineAsync(device, REDUCE_BANDS_WGSL, "main", [
        { binding: 1, type: ro },
        { binding: 2, type: rw },
      ]),
    ]);

    return {
      bglDyn0,
      stationExp,
      zeroPass,
      uqZero,
      uSetup,
      buildAbRow,
      modePairs,
      buildAbY,
      splitToOd,
      accHome,
      reduceU,
      reduceQ,
      reduceOzLen,
      lenPhase1,
      lenPhase2,
      slowPrepU,
      slowScatterU,
      slowSortU,
      slowBktCount,
      slowBktPrefix1,
      slowBktPrefix2,
      slowBktScatter,
      slowApplyOd,
      slowApplyQ,
      slowApplyHome,
      reduceBands,
    };
  }

  private createUploadedBuffer(
    label: string,
    usage: GPUBufferUsageFlags,
    src: ArrayBufferView,
  ): GPUBuffer {
    const size = Math.max(16, align4(src.byteLength));
    const buf = this.device.createBuffer({ label, size, usage: usage | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buf, 0, src.buffer, src.byteOffset, src.byteLength);
    return buf;
  }

  private ensureNetwork(
    apiKey: number,
    netEpoch: number,
    descriptors: Uint32Array,
    meta: Uint32Array,
    wasmMem: WebAssembly.Memory,
    neededMaxVBatch: number,
  ): ResidentNetworkBuffers {
    const key = `${apiKey}:${netEpoch}`;
    let entry = this.netCache.get(key);
    this.tick += 1;

    if (entry) {
      entry.lastUsedTick = this.tick;
      if (neededMaxVBatch > entry.maxVBatchAlloc) {
        const layout = computeBufferLayout(meta, neededMaxVBatch);
        this.assertBufferLimit("batch_f32", Math.max(256, layout.batchF32Floats * 4));
        entry.batchF32Buf.destroy();
        entry.layout = layout;
        entry.maxVBatchAlloc = neededMaxVBatch;
        entry.batchF32Buf = this.device.createBuffer({
          label: "batch_f32",
          size: Math.max(256, layout.batchF32Floats * 4),
          usage: GPUBufferUsage.STORAGE,
        });
      }
      return entry;
    }

    while (this.netCache.size >= MAX_NET_CACHE) {
      let oldestKey: string | null = null;
      let oldestTick = Infinity;
      for (const [k, v] of this.netCache) {
        if (v.lastUsedTick < oldestTick) {
          oldestTick = v.lastUsedTick;
          oldestKey = k;
        }
      }
      if (!oldestKey) break;
      const old = this.netCache.get(oldestKey)!;
      this.destroyNetworkEntry(old);
      this.netCache.delete(oldestKey);
    }

    const netF32Ptr = descriptors[3];
    const netF32Len = descriptors[4];
    const netU32Ptr = descriptors[5];
    const netU32Len = descriptors[6];

    const buf = wasmMem.buffer;
    const netF32View = new Float32Array(buf, netF32Ptr, netF32Len);
    const netU32View = new Uint32Array(buf, netU32Ptr, netU32Len);

    const nBatches = meta[7];
    const offActiveAOff = meta[32];
    const batchActiveALoHi = new Uint32Array(nBatches * 2);
    for (let b = 0; b < nBatches; b++) {
      batchActiveALoHi[b * 2] = netU32View[offActiveAOff + b];
      batchActiveALoHi[b * 2 + 1] = netU32View[offActiveAOff + b + 1];
    }

    const maxV = Math.max(1, neededMaxVBatch);
    const layout = computeBufferLayout(meta, maxV);
    const s = meta[0];
    const ss = Math.max(1, s * s);
    const skWalkWords = Math.ceil(ss / 2);

    const netF32Buf = this.createUploadedBuffer("net_f32", GPUBufferUsage.STORAGE, netF32View);
    const netU32Buf = this.createUploadedBuffer("net_u32", GPUBufferUsage.STORAGE, netU32View);
    const erBuf = this.device.createBuffer({
      label: "er_buf",
      size: Math.max(16, layout.erFloats * 4),
      usage: GPUBufferUsage.STORAGE,
    });
    const odBuf = this.device.createBuffer({
      label: "od_buf",
      size: Math.max(16, layout.odFloats * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const outScalarsBuf = this.device.createBuffer({
      label: "out_scalars",
      size: Math.max(16, layout.outScalarsFloats * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const skimF32Buf = this.device.createBuffer({
      label: "skim_f32",
      size: Math.max(16, ss * 2 * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const skimU32Buf = this.device.createBuffer({
      label: "skim_u32",
      size: Math.max(16, skWalkWords * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const batchF32Buf = this.device.createBuffer({
      label: "batch_f32",
      size: Math.max(256, layout.batchF32Floats * 4),
      usage: GPUBufferUsage.STORAGE,
    });
    const batchU32Buf = this.device.createBuffer({
      label: "batch_u32",
      size: Math.max(256, layout.batchU32Words * 4),
      usage: GPUBufferUsage.STORAGE,
    });
    const uniformSlots = nBatches + 1;
    const uniformBuf = this.device.createBuffer({
      label: "pass_uniforms",
      size: uniformSlots * 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const stagingOdBuf = this.device.createBuffer({
      label: "staging_od",
      size: Math.max(16, layout.odFloats * 4),
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const stagingScalarsBuf = this.device.createBuffer({
      label: "staging_scalars",
      size: Math.max(16, layout.outScalarsFloats * 4),
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    entry = {
      key,
      meta: new Uint32Array(meta),
      layout,
      batchActiveALoHi,
      netF32Buf,
      netU32Buf,
      erBuf,
      odBuf,
      outScalarsBuf,
      skimF32Buf,
      skimU32Buf,
      batchF32Buf,
      batchU32Buf,
      uniformBuf,
      stagingOdBuf,
      stagingScalarsBuf,
      maxVBatchAlloc: maxV,
      uniformSlotsAlloc: uniformSlots,
      approxBytes:
        netF32View.byteLength +
        netU32View.byteLength +
        (layout.erFloats + layout.odFloats * 2 + layout.outScalarsFloats * 2 + layout.batchF32Floats) * 4 +
        layout.batchU32Words * 4,
      lastUsedTick: this.tick,
    };
    this.netCache.set(key, entry);
    return entry;
  }

  private ensurePeriod(
    apiKey: number,
    netEpoch: number,
    periodIdx: number,
    descriptors: Uint32Array,
    meta: Uint32Array,
    wasmMem: WebAssembly.Memory,
  ): ResidentPeriodBuffers {
    const key = `${apiKey}:${netEpoch}:${periodIdx}`;
    const existing = this.periodCache.get(key);
    if (existing) {
      this.periodCache.delete(key);
      this.periodCache.set(key, existing);
      return existing;
    }

    while (this.periodCache.size >= MAX_PERIOD_CACHE) {
      const oldestKey = this.periodCache.keys().next().value;
      if (oldestKey === undefined) break;
      const old = this.periodCache.get(oldestKey)!;
      old.perF32Buf.destroy();
      old.perU32Buf.destroy();
      this.periodCache.delete(oldestKey);
    }

    const nVisits = descriptors[2];
    const visitF32Ptr = descriptors[7];
    const visitF32Len = descriptors[8];
    const visitU32Ptr = descriptors[9];
    const visitU32Len = descriptors[10];
    const ouPtr = descriptors[11];
    const ouLen = descriptors[12];
    const ozOffPtr = descriptors[13];
    const ozOffLen = descriptors[14];

    const buf = wasmMem.buffer;
    const visitF32 = new Float32Array(buf, visitF32Ptr, visitF32Len);
    const ouPeriod = new Float32Array(buf, ouPtr, ouLen);
    const perF32Packed = new Float32Array(visitF32Len + ouLen);
    perF32Packed.set(visitF32, 0);
    perF32Packed.set(ouPeriod, visitF32Len);

    const visitU32 = new Uint32Array(buf, visitU32Ptr, visitU32Len);
    const ozVisitOff = new Uint32Array(buf, ozOffPtr, ozOffLen);
    const perU32Packed = new Uint32Array(visitU32Len + ozOffLen);
    perU32Packed.set(visitU32, 0);
    perU32Packed.set(ozVisitOff, visitU32Len);

    const nBatches = meta[7];
    const offBatchMeta = meta[31];
    const netU32View = new Uint32Array(buf, descriptors[5], descriptors[6]);
    const batchVRange = new Uint32Array(nBatches * 2);
    let maxVBatch = 1;
    for (let b = 0; b < nBatches; b++) {
      const ozFirst = netU32View[offBatchMeta + b * 4 + 2];
      const ozLast = netU32View[offBatchMeta + b * 4 + 3];
      const vStart = ozVisitOff[ozFirst] ?? 0;
      const vEnd = ozVisitOff[ozLast + 1] ?? vStart;
      batchVRange[b * 2] = vStart;
      batchVRange[b * 2 + 1] = vEnd;
      if (vEnd > vStart) {
        maxVBatch = Math.max(maxVBatch, vEnd - vStart);
      }
    }

    const perF32Buf = this.createUploadedBuffer("per_f32", GPUBufferUsage.STORAGE, perF32Packed);
    const perU32Buf = this.createUploadedBuffer("per_u32", GPUBufferUsage.STORAGE, perU32Packed);

    const res: ResidentPeriodBuffers = {
      key,
      nVisits,
      offOuPeriod: visitF32Len,
      offOzVisitOff: visitU32Len,
      batchVRange,
      maxVBatch,
      perF32Buf,
      perU32Buf,
      approxBytes: perF32Packed.byteLength + perU32Packed.byteLength,
    };
    this.periodCache.set(key, res);
    return res;
  }

  private destroyNetworkEntry(n: ResidentNetworkBuffers): void {
    n.netF32Buf.destroy();
    n.netU32Buf.destroy();
    n.erBuf.destroy();
    n.odBuf.destroy();
    n.outScalarsBuf.destroy();
    n.skimF32Buf.destroy();
    n.skimU32Buf.destroy();
    n.batchF32Buf.destroy();
    n.batchU32Buf.destroy();
    n.uniformBuf.destroy();
    n.stagingOdBuf.destroy();
    n.stagingScalarsBuf.destroy();
  }

  destroyAllCaches(): void {
    for (const n of this.netCache.values()) {
      this.destroyNetworkEntry(n);
    }
    this.netCache.clear();
    for (const p of this.periodCache.values()) {
      p.perF32Buf.destroy();
      p.perU32Buf.destroy();
    }
    this.periodCache.clear();
  }

  invalidateApi(apiKey: number): void {
    const prefix = `${apiKey}:`;
    for (const [k, v] of this.netCache) {
      if (k.startsWith(prefix)) {
        this.destroyNetworkEntry(v);
        this.netCache.delete(k);
      }
    }
    for (const [k, v] of this.periodCache) {
      if (k.startsWith(prefix)) {
        v.perF32Buf.destroy();
        v.perU32Buf.destroy();
        this.periodCache.delete(k);
      }
    }
  }

  private assertBufferLimit(label: string, bytes: number): void {
    const maxBinding = this.device.limits.maxStorageBufferBindingSize;
    const maxBuffer = this.device.limits.maxBufferSize;
    if (bytes > maxBinding || bytes > maxBuffer) {
      throw new Error(
        `WebGPU buffer '${label}' (${bytes} bytes) exceeds device limit (maxStorage=${maxBinding}, maxBuffer=${maxBuffer})`,
      );
    }
  }

  async runModeChoicePass(inputs: GpuPassStageInputs): Promise<GpuPassResult> {
    if (this.disposed || this.brokenReason) {
      return { ok: false, reason: this.brokenReason ?? "WebGPU context disposed" };
    }
    const { apiKey, descriptors, netMeta, passUniforms, wasmMemory } = inputs;
    if (descriptors.length < 21 || netMeta.length < 37 || passUniforms.length < 64) {
      return { ok: false, reason: "invalid GPU staging descriptor length" };
    }

    const netEpoch = descriptors[0];
    const periodIdx = descriptors[1];
    const s = netMeta[0];
    const nb = netMeta[1];
    const nq = netMeta[2];
    const nu = netMeta[3];
    const naEntries = netMeta[6];
    const nBatches = netMeta[7];
    const parted = netMeta[8] !== 0;
    const maxAccSpan = netMeta.length >= 38 ? netMeta[37] : 64;

    if (s === 0 || nu === 0 || nq === 0 || nBatches === 0) {
      return { ok: false, reason: "empty network dimensions" };
    }

    const maxWg = this.device.limits.maxComputeWorkgroupsPerDimension;
    if (s > maxWg || Math.ceil(nq / 64) > maxWg) {
      return { ok: false, reason: `dimension exceeds maxComputeWorkgroupsPerDimension (${maxWg})` };
    }

    try {
      const preLayout = computeBufferLayout(netMeta, 1);
      const preSs = Math.max(1, s * s);
      const skWalkWords = Math.ceil(preSs / 2);
      this.assertBufferLimit("net_f32", descriptors[4] * 4);
      this.assertBufferLimit("net_u32", descriptors[6] * 4);
      this.assertBufferLimit("per_f32", (descriptors[8] + descriptors[12]) * 4);
      this.assertBufferLimit("per_u32", (descriptors[10] + descriptors[14]) * 4);
      this.assertBufferLimit("er_buf", preLayout.erFloats * 4);
      this.assertBufferLimit("od_buf", preLayout.odFloats * 4);
      this.assertBufferLimit("out_scalars", preLayout.outScalarsFloats * 4);
      this.assertBufferLimit("skim_f32", (preSs + s) * 4);
      this.assertBufferLimit("skim_u32", skWalkWords * 4);
      this.assertBufferLimit("batch_u32", preLayout.batchU32Words * 4);
      this.assertBufferLimit("batch_f32", preLayout.batchF32Floats * 4);

      const per = this.ensurePeriod(apiKey, netEpoch, periodIdx, descriptors, netMeta, wasmMemory);
      if (Math.ceil(per.maxVBatch / 64) > maxWg) {
        return { ok: false, reason: `maxVBatch exceeds maxComputeWorkgroupsPerDimension (${maxWg})` };
      }
      const postLayout = computeBufferLayout(netMeta, Math.max(1, per.maxVBatch));
      this.assertBufferLimit("batch_f32", postLayout.batchF32Floats * 4);

      const net = this.ensureNetwork(apiKey, netEpoch, descriptors, netMeta, wasmMemory, per.maxVBatch);

      // Upload dynamic skims from live WASM memory
      const buf = wasmMemory.buffer;
      const skRView = new Float32Array(buf, descriptors[15], descriptors[16]);
      const skWalkU32View = new Uint32Array(buf, descriptors[17], descriptors[18]);
      const skWalkMaxView = new Float32Array(buf, descriptors[19], descriptors[20]);

      this.device.queue.writeBuffer(net.skimF32Buf, 0, skRView.buffer, skRView.byteOffset, skRView.byteLength);
      this.device.queue.writeBuffer(
        net.skimF32Buf,
        s * s * 4,
        skWalkMaxView.buffer,
        skWalkMaxView.byteOffset,
        skWalkMaxView.byteLength,
      );
      this.device.queue.writeBuffer(
        net.skimU32Buf,
        0,
        skWalkU32View.buffer,
        skWalkU32View.byteOffset,
        skWalkU32View.byteLength,
      );

      // Build packed 256-byte uniform slots (slot 0 = global, slot 1..nBatches = batch b)
      const uniformBytes = new Uint8Array((nBatches + 1) * 256);
      const uF32 = new Float32Array(uniformBytes.buffer);
      const uU32 = new Uint32Array(uniformBytes.buffer);

      const writeSlot = (
        slot: number,
        batchIdx: number,
        vStart: number,
        vEnd: number,
        aLo: number,
        aHi: number,
      ) => {
        const base = slot * 64;
        uF32.set(passUniforms.subarray(0, 64), base);
        uU32[base + 54] = per.nVisits;
        uU32[base + 55] = per.offOuPeriod;
        uU32[base + 56] = per.offOzVisitOff;
        uU32[base + 57] = s * s;
        uU32[base + 58] = batchIdx;
        uU32[base + 59] = vStart;
        uU32[base + 60] = vEnd;
        uU32[base + 61] = aLo;
        uU32[base + 62] = aHi;
        uU32[base + 63] = per.maxVBatch;
      };

      writeSlot(0, 0, 0, 0, 0, 0);
      for (let b = 0; b < nBatches; b++) {
        writeSlot(
          1 + b,
          b,
          per.batchVRange[b * 2],
          per.batchVRange[b * 2 + 1],
          net.batchActiveALoHi[b * 2],
          net.batchActiveALoHi[b * 2 + 1],
        );
      }
      this.device.queue.writeBuffer(net.uniformBuf, 0, uniformBytes);

      const makeBg = (pipeline: GPUComputePipeline, buffers: GPUBuffer[]) =>
        this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: net.uniformBuf, offset: 0, size: 256 } },
            ...buffers.map((buffer, idx) => ({ binding: idx + 1, resource: { buffer } })),
          ],
        });

      const p = this.pipelines;
      const bgStationExp = makeBg(p.stationExp, [net.netF32Buf, net.skimF32Buf, net.erBuf]);
      const bgZeroPass = makeBg(p.zeroPass, [net.odBuf, net.outScalarsBuf, net.batchF32Buf]);
      const bgUqZero = makeBg(p.uqZero, [net.batchF32Buf, net.batchU32Buf]);
      const bgUSetup = makeBg(p.uSetup, [
        net.netF32Buf,
        net.netU32Buf,
        per.perF32Buf,
        per.perU32Buf,
        net.skimF32Buf,
        net.batchF32Buf,
        net.batchU32Buf,
      ]);
      const bgBuildAbRow = makeBg(p.buildAbRow, [
        net.netF32Buf,
        net.netU32Buf,
        net.erBuf,
        net.batchF32Buf,
        net.batchU32Buf,
      ]);
      const bgModePairs = makeBg(p.modePairs, [
        net.netF32Buf,
        net.netU32Buf,
        per.perF32Buf,
        net.erBuf,
        net.skimU32Buf,
        net.batchF32Buf,
        net.batchU32Buf,
      ]);
      const bgBuildAbY = makeBg(p.buildAbY, [net.netF32Buf, net.netU32Buf, net.batchF32Buf, net.batchU32Buf]);
      const bgSplitToOd = makeBg(p.splitToOd, [
        net.netF32Buf,
        net.netU32Buf,
        net.erBuf,
        net.batchF32Buf,
        net.batchU32Buf,
        net.odBuf,
      ]);
      const bgAccHome = makeBg(p.accHome, [
        net.netF32Buf,
        net.netU32Buf,
        net.erBuf,
        net.batchF32Buf,
        net.batchU32Buf,
        net.outScalarsBuf,
      ]);
      const bgReduceU = makeBg(p.reduceU, [net.netU32Buf, net.batchF32Buf, net.outScalarsBuf]);
      const bgReduceQ = makeBg(p.reduceQ, [
        net.netF32Buf,
        net.netU32Buf,
        net.batchF32Buf,
        net.outScalarsBuf,
      ]);
      const bgReduceOzLen = makeBg(p.reduceOzLen, [net.netU32Buf, per.perU32Buf, net.batchF32Buf]);
      const bgLenPhase1 = makeBg(p.lenPhase1, [
        net.netF32Buf,
        net.netU32Buf,
        per.perU32Buf,
        net.batchF32Buf,
      ]);
      const bgLenPhase2 = makeBg(p.lenPhase2, [net.batchF32Buf, net.outScalarsBuf]);
      const bgSlowBin = makeBg(p.slowPrepU, [net.batchU32Buf, net.outScalarsBuf]);
      const bgSlowApplyOd = makeBg(p.slowApplyOd, [net.batchU32Buf, net.odBuf]);
      const bgSlowApplyRest = makeBg(p.slowApplyQ, [net.batchU32Buf, net.outScalarsBuf]);
      const bgReduceBands = makeBg(p.reduceBands, [net.netU32Buf, net.outScalarsBuf]);

      const ss = s * s;
      const offBatchMeta = netMeta[31];
      const netU32Host = new Uint32Array(buf, descriptors[5], descriptors[6]);

      let scopesPushed = 0;
      let querySet: GPUQuerySet | null = null;
      let tsResolveBuf: GPUBuffer | null = null;
      let tsReadBuf: GPUBuffer | null = null;
      let odIsMapped = false;
      let scIsMapped = false;
      let tsIsMapped = false;

      try {
        this.device.pushErrorScope("validation");
        scopesPushed++;
        this.device.pushErrorScope("out-of-memory");
        scopesPushed++;

        const encoder = this.device.createCommandEncoder();
        const tsEntries: Array<{ name: string; slot: number }> = [];
        if (this.hasTimestamps) {
          querySet = this.device.createQuerySet({ type: "timestamp", count: 4096 });
          tsResolveBuf = this.device.createBuffer({
            size: 4096 * 8,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
          });
          tsReadBuf = this.device.createBuffer({
            size: 4096 * 8,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
          });
        }
        let singlePass: GPUComputePassEncoder | null = this.hasTimestamps
          ? null
          : encoder.beginComputePass();

        const dispatchK = (
          name: string,
          pipeline: GPUComputePipeline,
          bg: GPUBindGroup,
          dyn: number[],
          x: number,
          y = 1,
          z = 1,
        ) => {
          if (this.hasTimestamps && querySet && tsEntries.length * 2 + 1 < 4096) {
            const slot = tsEntries.length * 2;
            tsEntries.push({ name, slot });
            const cp = encoder.beginComputePass({
              timestampWrites: {
                querySet,
                beginningOfPassWriteIndex: slot,
                endOfPassWriteIndex: slot + 1,
              },
            });
            cp.setPipeline(pipeline);
            cp.setBindGroup(0, bg, dyn);
            cp.dispatchWorkgroups(x, y, z);
            cp.end();
          } else if (singlePass) {
            singlePass.setPipeline(pipeline);
            singlePass.setBindGroup(0, bg, dyn);
            singlePass.dispatchWorkgroups(x, y, z);
          }
        };

        const dispatch1D = (
          name: string,
          pipeline: GPUComputePipeline,
          bg: GPUBindGroup,
          dyn: number[],
          totalWg: number,
        ) => {
          if (totalWg <= 0) return;
          const x = Math.min(totalWg, maxWg);
          const y = Math.ceil(totalWg / x);
          dispatchK(name, pipeline, bg, dyn, x, y);
        };

        // 1. Pass initialization: er_buf & zero outputs
        const zeroCount = Math.max(ss, net.layout.outScalarsFloats, net.layout.lenPartFloats);
        dispatch1D("zeroPass", p.zeroPass, bgZeroPass, [0], Math.ceil(zeroCount / 256));
        dispatch1D("stationExp", p.stationExp, bgStationExp, [0], Math.ceil((nb * ss) / 256));

        // 2. Origin batches in canonical ascending u_ord order
        const uqLen = GPU_U_BATCH * nq * 10;
        const nqWg = Math.ceil(nq / 64);
        const sWg = Math.ceil(s / 64);
        const lenPartWg = Math.ceil(W_LEN / 64);
        const lenSlotsWg = Math.ceil((6 * N_LEN) / 64);
        const ruleEnabled =
          new Uint32Array(passUniforms.buffer, passUniforms.byteOffset, passUniforms.length)[9] !== 0;

        for (let b = 0; b < nBatches; b++) {
          const dyn = [(1 + b) * 256];
          const uStart = netU32Host[offBatchMeta + b * 4 + 0];
          const uEnd = netU32Host[offBatchMeta + b * 4 + 1];
          const batchNu = uEnd - uStart;
          const vCnt = per.batchVRange[b * 2 + 1] - per.batchVRange[b * 2];
          const activeACnt = net.batchActiveALoHi[b * 2 + 1] - net.batchActiveALoHi[b * 2];

          const clearWords = Math.max(uqLen, 324, vCnt * 3);
          dispatch1D("uqZero", p.uqZero, bgUqZero, dyn, Math.ceil(clearWords / 256));
          dispatchK("uSetup", p.uSetup, bgUSetup, dyn, batchNu);
          dispatchK("buildAbRow", p.buildAbRow, bgBuildAbRow, dyn, sWg, nb, batchNu);
          dispatchK("modePairs", p.modePairs, bgModePairs, dyn, nqWg, batchNu);
          dispatchK("buildAbY", p.buildAbY, bgBuildAbY, dyn, 1, s, 2 * nb);

          if (activeACnt > 0) {
            dispatchK("splitToOd", p.splitToOd, bgSplitToOd, dyn, sWg, activeACnt);
          }

          if (maxAccSpan > 0) {
            dispatchK("accHome", p.accHome, bgAccHome, dyn, Math.min(maxAccSpan, maxWg), batchNu);
          }
          dispatchK("reduceU", p.reduceU, bgReduceU, dyn, batchNu);
          dispatchK("reduceQ", p.reduceQ, bgReduceQ, dyn, nqWg);

          if (vCnt > 0) {
            dispatchK("reduceOzLen", p.reduceOzLen, bgReduceOzLen, dyn, Math.ceil(vCnt / 64));
            dispatchK("lenPhase1", p.lenPhase1, bgLenPhase1, dyn, lenPartWg);
          }

          // Deterministic slow-pair sort + apply (when walking rule is enabled)
          if (ruleEnabled) {
            dispatchK("slowPrepU", p.slowPrepU, bgSlowBin, dyn, 1);
            dispatchK("slowScatterU", p.slowScatterU, bgSlowBin, dyn, 1);
            dispatchK("slowSortU", p.slowSortU, bgSlowBin, dyn, batchNu);
            dispatchK("slowBktCount", p.slowBktCount, bgSlowBin, dyn, 8, 3);
            dispatchK("slowBktPrefix1", p.slowBktPrefix1, bgSlowBin, dyn, 64, 3);
            dispatchK("slowBktPrefix2", p.slowBktPrefix2, bgSlowBin, dyn, 1, 3);
            dispatchK("slowBktScatter", p.slowBktScatter, bgSlowBin, dyn, 8, 3);
            dispatchK("slowApplyOd", p.slowApplyOd, bgSlowApplyOd, dyn, 64);
            dispatchK("slowApplyQ", p.slowApplyQ, bgSlowApplyRest, dyn, 64);
            dispatchK("slowApplyHome", p.slowApplyHome, bgSlowApplyRest, dyn, 64);
          }
        }

        // 3. Final reductions: trip-length partial sum & distance-band aggregation (`rail_band[0..12]`)
        dispatchK("lenPhase2", p.lenPhase2, bgLenPhase2, [0], lenSlotsWg);
        dispatchK("reduceBands", p.reduceBands, bgReduceBands, [0], N_BANDS);

        if (singlePass) {
          singlePass.end();
        }
        if (querySet && tsResolveBuf && tsReadBuf && tsEntries.length > 0) {
          encoder.resolveQuerySet(querySet, 0, tsEntries.length * 2, tsResolveBuf, 0);
          encoder.copyBufferToBuffer(tsResolveBuf, 0, tsReadBuf, 0, tsEntries.length * 2 * 8);
        }

        // 4. Single staging readback copy
        encoder.copyBufferToBuffer(net.odBuf, 0, net.stagingOdBuf, 0, ss * 4);
        encoder.copyBufferToBuffer(
          net.outScalarsBuf,
          0,
          net.stagingScalarsBuf,
          0,
          net.layout.outScalarsFloats * 4,
        );

        this.device.queue.submit([encoder.finish()]);

        await Promise.all([
          net.stagingOdBuf.mapAsync(GPUMapMode.READ).then(() => {
            odIsMapped = true;
          }),
          net.stagingScalarsBuf.mapAsync(GPUMapMode.READ).then(() => {
            scIsMapped = true;
          }),
          tsReadBuf
            ? tsReadBuf.mapAsync(GPUMapMode.READ).then(() => {
                tsIsMapped = true;
              })
            : Promise.resolve(),
        ]);

        let kernelMs: Record<string, number> | undefined;
        if (tsReadBuf && tsIsMapped) {
          const tsU64 = new BigUint64Array(tsReadBuf.getMappedRange());
          kernelMs = {};
          for (const { name, slot } of tsEntries) {
            const dt = Number(tsU64[slot + 1] - tsU64[slot]) / 1e6;
            kernelMs[name] = (kernelMs[name] ?? 0) + dt;
          }
        }

        const scopesToPop = scopesPushed;
        scopesPushed = 0;
        const popped: Array<GPUError | null> = [];
        for (let i = 0; i < scopesToPop; i++) {
          popped.push(await this.device.popErrorScope().catch(() => null));
        }
        const oomErr = popped[0] ?? null;
        const valErr = popped[1] ?? null;

        if (oomErr || valErr) {
          const reason = valErr?.message ?? oomErr?.message ?? "WebGPU error";
          this.brokenReason = reason;
          this.dispose();
          return { ok: false, reason };
        }

        const odMapped = new Float32Array(net.stagingOdBuf.getMappedRange());
        const scMapped = new Float32Array(net.stagingScalarsBuf.getMappedRange());

        const o = net.layout.outOffsets;
        const overflowFlag = scMapped[o.status];
        if (overflowFlag > 0.5) {
          return {
            ok: false,
            reason: `slow_count exceeded MAX_SLOW (${scMapped[o.status + 1]})`,
          };
        }

        const od = new Float32Array(odMapped);
        const accHome = scMapped.slice(o.accHome, o.accHome + naEntries);
        const accQFlat = scMapped.slice(o.accQ, o.accQ + nq * KE);
        const railBySub = scMapped.slice(o.railBySub, o.railBySub + nu);
        const walkLostBySub = scMapped.slice(o.walkLostBySub, o.walkLostBySub + nu);
        const unrLostBySub = parted
          ? scMapped.slice(o.unrLostBySub, o.unrLostBySub + nu)
          : new Float32Array(0);
        const uSense = scMapped.slice(o.uSense, o.uSense + nu * 3);
        const lostQ = scMapped.slice(o.lostQ, o.lostQ + nq);
        const unrQ = parted
          ? scMapped.slice(o.unrQ, o.unrQ + nq)
          : new Float32Array(0);
        const railBand = scMapped.slice(o.railBand, o.railBand + N_BANDS);
        const lenFlat = scMapped.slice(o.lenOut, o.lenOut + 6 * N_LEN);

        return {
          ok: true,
          kernelMs,
          outputs: {
            od,
            accHome,
            accQFlat,
            railBySub,
            walkLostBySub,
            unrLostBySub,
            uSense,
            lostQ,
            unrQ,
            railBand,
            lenFlat,
          },
        };
      } finally {
        if (odIsMapped || (!this.disposed && net.stagingOdBuf.mapState !== "unmapped")) {
          try {
            net.stagingOdBuf.unmap();
          } catch {
            // ignore unmap on destroyed device
          }
        }
        if (scIsMapped || (!this.disposed && net.stagingScalarsBuf.mapState !== "unmapped")) {
          try {
            net.stagingScalarsBuf.unmap();
          } catch {
            // ignore unmap on destroyed device
          }
        }
        if (tsReadBuf) {
          if (tsIsMapped || (!this.disposed && tsReadBuf.mapState !== "unmapped")) {
            try {
              tsReadBuf.unmap();
            } catch {
              // ignore
            }
          }
          try {
            tsReadBuf.destroy();
          } catch {
            // ignore
          }
        }
        try {
          tsResolveBuf?.destroy();
        } catch {
          // ignore
        }
        try {
          querySet?.destroy();
        } catch {
          // ignore
        }
        while (scopesPushed > 0) {
          scopesPushed--;
          await this.device.popErrorScope().catch(() => null);
        }
      }
    } catch (err) {
      return {
        ok: false,
        reason: err instanceof Error ? err.message : String(err),
      };
    }
  }
}
