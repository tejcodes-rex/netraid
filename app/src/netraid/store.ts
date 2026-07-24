// Encrypted local storage (op-sqlite + SQLCipher). Stores ONLY embeddings and
// attendance events, never raw face images. AES-256 at rest; key from keychain.

import { open, type DB } from '@op-engineering/op-sqlite';
import type { AttendanceRecord, EnrolledTemplate, Embedding } from './types';
import { getOrCreateDbKey } from './keys';

let db: DB | null = null;

export async function initStore(): Promise<void> {
  if (db) return;
  const encryptionKey = await getOrCreateDbKey();
  db = open({ name: 'netraid.db', encryptionKey });

  db.executeSync(`CREATE TABLE IF NOT EXISTS templates (
    person_id  TEXT PRIMARY KEY,
    embedding  BLOB NOT NULL,
    dims       INTEGER NOT NULL,
    created_at INTEGER NOT NULL
  );`);

  db.executeSync(`CREATE TABLE IF NOT EXISTS attendance (
    id          TEXT PRIMARY KEY,
    person_id   TEXT NOT NULL,
    ts          INTEGER NOT NULL,
    site_id     TEXT,
    lat REAL, lng REAL,
    device_id   TEXT NOT NULL,
    liveness    INTEGER NOT NULL,
    score       REAL NOT NULL,
    sync_state  TEXT NOT NULL DEFAULT 'pending',
    attempts    INTEGER NOT NULL DEFAULT 0,
    created_at  INTEGER NOT NULL
  );`);
  db.executeSync(
    `CREATE INDEX IF NOT EXISTS idx_att_sync ON attendance(sync_state, ts);`,
  );
}

function bufFromFloats(v: Float32Array): ArrayBuffer {
  const ab = new ArrayBuffer(v.byteLength);
  new Float32Array(ab).set(v);
  return ab;
}
function floatsFromBuf(b: ArrayBuffer): Float32Array {
  return new Float32Array(b);
}

// NOTE: all data-path queries use executeSync, not the async execute(). On some
// Android JSI builds (seen on Vivo) the async execute() promise never resolves,
// so an enroll Save would hang forever. The synchronous path is proven to work
// (it is what initStore uses for table creation) and these rows are tiny, so the
// main-thread cost is negligible.
export async function saveTemplate(personId: string, embedding: Embedding) {
  db!.executeSync(
    `INSERT OR REPLACE INTO templates (person_id, embedding, dims, created_at)
     VALUES (?, ?, ?, ?)`,
    [personId, bufFromFloats(embedding), embedding.length, Date.now()],
  );
}

export async function loadTemplates(): Promise<EnrolledTemplate[]> {
  const { rows } = db!.executeSync(
    `SELECT person_id, embedding, created_at FROM templates`,
  );
  return (rows ?? []).map((r: any) => ({
    personId: r.person_id,
    embedding: floatsFromBuf(r.embedding as ArrayBuffer),
    createdAt: r.created_at,
  }));
}

export async function countTemplates(): Promise<number> {
  const { rows } = db!.executeSync(`SELECT COUNT(*) AS n FROM templates`);
  return (rows?.[0] as any)?.n ?? 0;
}

export async function countPendingAttendance(): Promise<number> {
  const { rows } = db!.executeSync(
    `SELECT COUNT(*) AS n FROM attendance WHERE sync_state='pending'`,
  );
  return (rows?.[0] as any)?.n ?? 0;
}

export async function insertAttendance(rec: AttendanceRecord) {
  db!.executeSync(
    `INSERT OR IGNORE INTO attendance
     (id, person_id, ts, site_id, lat, lng, device_id, liveness, score, sync_state, attempts, created_at)
     VALUES (?,?,?,?,?,?,?,?,?,?,?,?)`,
    [
      rec.id, rec.personId, rec.ts, rec.siteId ?? null, rec.lat ?? null,
      rec.lng ?? null, rec.deviceId, rec.livenessPassed ? 1 : 0, rec.matchScore,
      rec.syncState, rec.attempts, rec.createdAt,
    ],
  );
}

export async function pendingAttendance(limit = 100): Promise<AttendanceRecord[]> {
  const { rows } = db!.executeSync(
    `SELECT * FROM attendance WHERE sync_state='pending' ORDER BY ts LIMIT ?`,
    [limit],
  );
  return (rows ?? []).map(rowToRecord);
}

export async function markSynced(ids: string[]) {
  if (!ids.length) return;
  const ph = ids.map(() => '?').join(',');
  db!.executeSync(
    `UPDATE attendance SET sync_state='synced' WHERE id IN (${ph})`, ids,
  );
}

export async function bumpAttempts(ids: string[]) {
  if (!ids.length) return;
  const ph = ids.map(() => '?').join(',');
  db!.executeSync(
    `UPDATE attendance SET attempts=attempts+1 WHERE id IN (${ph})`, ids,
  );
}

/** Purge ONLY confirmed-synced PII rows, then reclaim pages so data is overwritten. */
export async function purgeSynced(ids: string[]) {
  if (!ids.length) return;
  const ph = ids.map(() => '?').join(',');
  db!.executeSync(`DELETE FROM attendance WHERE id IN (${ph})`, ids);
  db!.executeSync('VACUUM');
}

function rowToRecord(r: any): AttendanceRecord {
  return {
    id: r.id, personId: r.person_id, ts: r.ts, siteId: r.site_id ?? undefined,
    lat: r.lat ?? undefined, lng: r.lng ?? undefined, deviceId: r.device_id,
    livenessPassed: !!r.liveness, matchScore: r.score, syncState: r.sync_state,
    attempts: r.attempts, createdAt: r.created_at,
  };
}
