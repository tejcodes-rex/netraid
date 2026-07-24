// Offline-first sync engine. Writes are always local-first; this flushes the
// pending queue to AWS when the internet is genuinely reachable, then purges the
// confirmed rows. Idempotent via per-record client UUIDs.

import NetInfo from '@react-native-community/netinfo';
import {
  pendingAttendance,
  markSynced,
  bumpAttempts,
  purgeSynced,
} from './store';
import { getDeviceId } from './keys';

// Configure via your RN env tooling (e.g. react-native-config). Falls back to a placeholder.
declare const process: { env?: Record<string, string | undefined> } | undefined;
const API_BASE =
  (typeof process !== 'undefined' && process?.env?.NETRAID_API_BASE) ||
  'https://api.netraid.nhai.example/v1';

let running = false;
let unsubscribe: (() => void) | null = null;

/** Start listening for connectivity and flush opportunistically. */
export function startSync(): void {
  if (unsubscribe) return;
  unsubscribe = NetInfo.addEventListener((state) => {
    // isInternetReachable guards against captive Wi-Fi with no real internet.
    if (state.isConnected && state.isInternetReachable) {
      void flush();
    }
  });
}

export function stopSync(): void {
  unsubscribe?.();
  unsubscribe = null;
}

/** Push all pending records, then purge the ones the server confirmed. */
export async function flush(): Promise<{ synced: number; failed: number }> {
  if (running) return { synced: 0, failed: 0 };
  running = true;
  let synced = 0;
  let failed = 0;
  try {
    const deviceId = await getDeviceId();
    // Loop in batches until the queue drains or a batch fails.
    for (;;) {
      const batch = await pendingAttendance(100);
      if (batch.length === 0) break;

      let res: Response;
      try {
        res = await fetch(`${API_BASE}/attendance/sync`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Device-Id': deviceId,
          // Authorization: `Bearer ${await getAccessToken()}` // Cognito JWT
        },
        body: JSON.stringify({
          records: batch.map((r) => ({
            id: r.id,
            personId: r.personId,
            ts: r.ts,
            siteId: r.siteId,
            lat: r.lat,
            lng: r.lng,
            deviceId: r.deviceId,
            livenessPassed: r.livenessPassed,
            matchScore: r.matchScore,
          })),
        }),
        });
      } catch {
        // Transport failure (endpoint unreachable, DNS, radio drop): the queue
        // stays intact and the next connectivity event retries. Never let this
        // escape as an unhandled rejection.
        await bumpAttempts(batch.map((r) => r.id));
        failed += batch.length;
        break;
      }

      if (!res.ok) {
        await bumpAttempts(batch.map((r) => r.id));
        failed += batch.length;
        break; // server/transport error, retry on next connectivity event
      }

      const { results } = (await res.json()) as {
        results: { id: string; status: 'ok' | 'rejected' }[];
      };
      const ok = results.filter((r) => r.status === 'ok').map((r) => r.id);
      const bad = results.filter((r) => r.status === 'rejected').map((r) => r.id);
      await markSynced(ok);
      await bumpAttempts(bad);
      await purgeSynced(ok); // purge ONLY confirmed rows, immediately after ACK
      synced += ok.length;
      failed += bad.length;
      if (batch.length < 100) break;
    }
  } catch {
    // Defensive: a malformed server response (bad JSON, unexpected shape) must
    // not crash the app or leak an unhandled rejection; records stay queued.
  } finally {
    running = false;
  }
  return { synced, failed };
}
