import { perfOff } from "../../perfFlags";
import { requestWebGpuAdapter, requestWebGpuDevice } from "./adapter";

export type WebGpuStatus =
  | "ready"
  | "fallback-adapter"
  | "browser-blocked"
  | "disabled"
  | "unsupported"
  | "no-adapter"
  | "limits"
  | "device-failed";

export interface WebGpuCapability {
  canRun: boolean;
  status: WebGpuStatus;
  adapterName: string | null;
  hardwareGpu: string | null;
  reason: string | null;
  maxStorageMiB: number;
}

const MIN_STORAGE_BINDING_BYTES = 32 * 1024 * 1024;
const MIN_STORAGE_BUFFERS_PER_STAGE = 8;
const MIN_WORKGROUP_INVOCATIONS = 256;
const MIN_WORKGROUP_STORAGE_BYTES = 16 * 1024;

const SOFTWARE_GPU_RE = /swiftshader|llvmpipe|softpipe|software|microsoft basic render/i;

let cachedReadyProbe: WebGpuCapability | null = null;
let inflightProbe: Promise<WebGpuCapability> | null = null;

function formatAdapterName(info?: GPUAdapterInfo): string | null {
  if (!info) return null;
  const parts = [info.description, info.architecture, info.device, info.vendor]
    .map((s) => (s ?? "").trim())
    .filter(Boolean);
  return parts[0] ?? null;
}

/**
 * Inspect WebGL2/WebGL for a non-software GPU renderer so we can tell when WebGPU is merely
 * disabled in the browser (e.g. Vulkan off on Linux Chromium/Brave or non-HTTPS origin) vs absent.
 */
function detectHardwareWebGlGpu(): string | null {
  if (typeof document === "undefined") return null;
  let canvas: HTMLCanvasElement | null = null;
  try {
    canvas = document.createElement("canvas");
    canvas.width = 1;
    canvas.height = 1;
    const gl = (canvas.getContext("webgl2") ?? canvas.getContext("webgl")) as
      | WebGL2RenderingContext
      | WebGLRenderingContext
      | null;
    if (!gl) return null;
    try {
      const dbg = gl.getExtension("WEBGL_debug_renderer_info");
      const raw = String(
        (dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER)) ?? "",
      ).trim();
      if (!raw || SOFTWARE_GPU_RE.test(raw)) return null;
      return raw.replace(/^ANGLE\s*\((.+)\)$/i, "$1").trim();
    } finally {
      gl.getExtension("WEBGL_lose_context")?.loseContext();
    }
  } catch {
    return null;
  } finally {
    canvas = null;
  }
}

/**
 * Probe whether this browser and GPU can run the WebGPU compute pass (`DemandGpuContext`).
 * Compiles a minimal compute module on a short-lived device and destroys it immediately so no VRAM stays held.
 */
export function probeWebGpuSupport(opts?: { fresh?: boolean }): Promise<WebGpuCapability> {
  if (!opts?.fresh && cachedReadyProbe) return Promise.resolve(cachedReadyProbe);
  if (!opts?.fresh && inflightProbe) return inflightProbe;
  const run = runProbe().then((res) => {
    if (res.canRun) cachedReadyProbe = res;
    if (inflightProbe === run) inflightProbe = null;
    return res;
  });
  inflightProbe = run;
  return run;
}

async function runProbe(): Promise<WebGpuCapability> {
  if (perfOff("webgpu")) {
    return {
      canRun: false,
      status: "disabled",
      adapterName: null,
      hardwareGpu: null,
      reason: "Disabled by ?perfOff=webgpu",
      maxStorageMiB: 0,
    };
  }
  if (typeof navigator === "undefined" || !("gpu" in navigator) || !navigator.gpu) {
    const hardwareGpu = detectHardwareWebGlGpu();
    const insecure = typeof window !== "undefined" && !window.isSecureContext;
    const reason = insecure ? "WebGPU requires HTTPS or localhost" : "WebGPU API unavailable";
    return {
      canRun: false,
      status: hardwareGpu ? "browser-blocked" : "unsupported",
      adapterName: null,
      hardwareGpu,
      reason,
      maxStorageMiB: 0,
    };
  }

  const { adapter, error: adapterError } = await requestWebGpuAdapter();
  if (!adapter) {
    const hardwareGpu = detectHardwareWebGlGpu();
    return {
      canRun: false,
      status: hardwareGpu ? "browser-blocked" : "no-adapter",
      adapterName: null,
      hardwareGpu,
      reason: adapterError ?? "No compatible WebGPU adapter",
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
      hardwareGpu: null,
      reason: "Adapter compute limits too low",
      maxStorageMiB,
    };
  }

  let device: GPUDevice | null = null;
  try {
    device = await requestWebGpuDevice(adapter, { maxStorageBindingMiB: 256, maxBufferMiB: 512 });
    const mod = device.createShaderModule({
      code: `@compute @workgroup_size(1) fn main() {}`,
    });
    if (typeof mod.getCompilationInfo === "function") {
      const info = await mod.getCompilationInfo();
      if (info.messages.some((m) => m.type === "error")) {
        return {
          canRun: false,
          status: "device-failed",
          adapterName,
          hardwareGpu: null,
          reason: "Compute shader compile failed",
          maxStorageMiB,
        };
      }
    }
    return {
      canRun: true,
      status: isFallback ? "fallback-adapter" : "ready",
      adapterName,
      hardwareGpu: null,
      reason: isFallback ? "Software WebGPU adapter" : null,
      maxStorageMiB,
    };
  } catch (err) {
    return {
      canRun: false,
      status: "device-failed",
      adapterName,
      hardwareGpu: null,
      reason: String(err),
      maxStorageMiB,
    };
  } finally {
    device?.destroy();
  }
}
