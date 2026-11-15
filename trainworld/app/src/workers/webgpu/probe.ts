import { perfOff } from "../../perfFlags";

export type WebGpuStatus =
  | "ready"
  | "fallback-adapter"
  | "disabled"
  | "unsupported"
  | "no-adapter"
  | "limits"
  | "device-failed";

export interface WebGpuCapability {
  canRun: boolean;
  status: WebGpuStatus;
  adapterName: string | null;
  reason: string | null;
  maxStorageMiB: number;
}

const MIN_STORAGE_BINDING_BYTES = 32 * 1024 * 1024;
const MIN_STORAGE_BUFFERS_PER_STAGE = 8;
const MIN_WORKGROUP_INVOCATIONS = 256;
const MIN_WORKGROUP_STORAGE_BYTES = 16 * 1024;

let cachedProbe: Promise<WebGpuCapability> | null = null;

function formatAdapterName(info?: GPUAdapterInfo): string | null {
  if (!info) return null;
  const parts = [info.description, info.architecture, info.device, info.vendor]
    .map((s) => (s ?? "").trim())
    .filter(Boolean);
  return parts[0] ?? null;
}

/**
 * Probe whether this browser and GPU can run the WebGPU compute pass (`DemandGpuContext`).
 * Compiles a minimal compute module on a short-lived device and destroys it immediately so no VRAM stays held.
 */
export function probeWebGpuSupport(opts?: { fresh?: boolean }): Promise<WebGpuCapability> {
  if (!opts?.fresh && cachedProbe) return cachedProbe;
  const run = runProbe();
  if (!opts?.fresh) cachedProbe = run;
  return run;
}

async function runProbe(): Promise<WebGpuCapability> {
  if (perfOff("webgpu")) {
    return {
      canRun: false,
      status: "disabled",
      adapterName: null,
      reason: "Disabled by ?perfOff=webgpu",
      maxStorageMiB: 0,
    };
  }
  if (typeof navigator === "undefined" || !("gpu" in navigator) || !navigator.gpu) {
    return {
      canRun: false,
      status: "unsupported",
      adapterName: null,
      reason: "WebGPU API unavailable",
      maxStorageMiB: 0,
    };
  }

  let adapter: GPUAdapter | null = null;
  try {
    adapter =
      (await navigator.gpu.requestAdapter({ powerPreference: "high-performance" })) ??
      (await navigator.gpu.requestAdapter());
  } catch (err) {
    return {
      canRun: false,
      status: "no-adapter",
      adapterName: null,
      reason: String(err),
      maxStorageMiB: 0,
    };
  }

  if (!adapter) {
    return {
      canRun: false,
      status: "no-adapter",
      adapterName: null,
      reason: "No compatible WebGPU adapter",
      maxStorageMiB: 0,
    };
  }

  const adapterName = formatAdapterName(adapter.info);
  const maxStorageMiB = Math.floor(adapter.limits.maxStorageBufferBindingSize / (1024 * 1024));
  const isFallback = !!adapter.info?.isFallbackAdapter;

  if (
    adapter.limits.maxStorageBufferBindingSize < MIN_STORAGE_BINDING_BYTES ||
    adapter.limits.maxStorageBuffersPerShaderStage < MIN_STORAGE_BUFFERS_PER_STAGE ||
    adapter.limits.maxComputeInvocationsPerWorkgroup < MIN_WORKGROUP_INVOCATIONS ||
    adapter.limits.maxComputeWorkgroupStorageSize < MIN_WORKGROUP_STORAGE_BYTES
  ) {
    return {
      canRun: false,
      status: "limits",
      adapterName,
      reason: "Adapter compute limits too low",
      maxStorageMiB,
    };
  }

  let device: GPUDevice | null = null;
  try {
    const requiredLimits: Record<string, number> = {};
    if (adapter.limits.maxStorageBufferBindingSize >= 128 * 1024 * 1024) {
      requiredLimits.maxStorageBufferBindingSize = Math.min(
        adapter.limits.maxStorageBufferBindingSize,
        256 * 1024 * 1024,
      );
    }
    if (adapter.limits.maxBufferSize >= 256 * 1024 * 1024) {
      requiredLimits.maxBufferSize = Math.min(
        adapter.limits.maxBufferSize,
        512 * 1024 * 1024,
      );
    }
    device = await adapter.requestDevice({ requiredLimits });
    const mod = device.createShaderModule({
      code: `@compute @workgroup_size(1) fn main() {}`,
    });
    const info = await mod.getCompilationInfo();
    if (info.messages.some((m) => m.type === "error")) {
      return {
        canRun: false,
        status: "device-failed",
        adapterName,
        reason: "Compute shader compile failed",
        maxStorageMiB,
      };
    }
    return {
      canRun: true,
      status: isFallback ? "fallback-adapter" : "ready",
      adapterName,
      reason: isFallback ? "Software WebGPU adapter" : null,
      maxStorageMiB,
    };
  } catch (err) {
    return {
      canRun: false,
      status: "device-failed",
      adapterName,
      reason: String(err),
      maxStorageMiB,
    };
  } finally {
    device?.destroy();
  }
}
