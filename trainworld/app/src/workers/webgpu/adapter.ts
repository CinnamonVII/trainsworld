/** Shared WebGPU adapter and device negotiation with safe fallbacks across browser quirks. */

export interface DeviceRequestOptions {
  maxStorageBindingMiB?: number;
  maxBufferMiB?: number;
  requiredFeatures?: GPUFeatureName[];
}

/**
 * Request a compatible `GPUAdapter`, trying `high-performance`, default, and `low-power` in turn
 * so a driver error or null response on one preference does not hide a working adapter.
 */
export async function requestWebGpuAdapter(): Promise<{ adapter: GPUAdapter | null; error: string | null }> {
  if (typeof navigator === "undefined" || !("gpu" in navigator) || !navigator.gpu) {
    return { adapter: null, error: "WebGPU API unavailable" };
  }
  const attempts: Array<GPURequestAdapterOptions | undefined> = [
    { powerPreference: "high-performance" },
    undefined,
    { powerPreference: "low-power" },
  ];
  let lastError: string | null = null;
  for (const opts of attempts) {
    try {
      const adapter = await navigator.gpu.requestAdapter(opts);
      if (adapter) return { adapter, error: null };
    } catch (err) {
      lastError = String(err);
    }
  }
  return { adapter: null, error: lastError };
}

/**
 * Request a `GPUDevice` with elevated buffer limits when supported, falling back to default
 * limits if a driver rejects explicit limit requests.
 */
export async function requestWebGpuDevice(
  adapter: GPUAdapter,
  opts?: DeviceRequestOptions,
): Promise<GPUDevice> {
  const requiredFeatures = opts?.requiredFeatures ?? [];
  const requiredLimits: Record<string, number> = {};
  const wantStorageBytes = (opts?.maxStorageBindingMiB ?? 256) * 1024 * 1024;
  const wantBufferBytes = (opts?.maxBufferMiB ?? 512) * 1024 * 1024;

  if (adapter.limits.maxStorageBufferBindingSize >= 128 * 1024 * 1024) {
    requiredLimits.maxStorageBufferBindingSize = Math.min(
      adapter.limits.maxStorageBufferBindingSize,
      wantStorageBytes,
    );
  }
  if (adapter.limits.maxBufferSize >= 256 * 1024 * 1024) {
    requiredLimits.maxBufferSize = Math.min(
      adapter.limits.maxBufferSize,
      wantBufferBytes,
    );
  }

  const hasLimits = Object.keys(requiredLimits).length > 0;
  try {
    return await adapter.requestDevice({
      ...(requiredFeatures.length ? { requiredFeatures } : {}),
      ...(hasLimits ? { requiredLimits } : {}),
    });
  } catch (err) {
    if (!hasLimits) throw err;
    return await adapter.requestDevice(requiredFeatures.length ? { requiredFeatures } : undefined);
  }
}
