# NetraID, Security & Privacy

Biometric authentication for a government workforce demands a high bar. NetraID is designed
**privacy-first** and aligned with India's **Digital Personal Data Protection (DPDP) Act,
2023**. This is engineering guidance, not legal advice, NHAI's compliance team should
confirm before production.

## 1. Threat model
| Threat | Mitigation |
|---|---|
| Attendance fraud (photo/screen/video) | Dual offline liveness (active challenge + passive MiniFASNet), see `LIVENESS.md` |
| Device theft → template extraction | Embeddings-only, AES-256 at rest, key in hardware keystore, biometric-gated |
| Network interception | TLS in transit; only attendance events + non-PII metadata sync; JWT auth |
| Replay of captured biometrics | Randomized challenge order + per-step timeout + server-side idempotency |
| Backend data breach | Embeddings/events only (no images); SSE at rest; least-privilege IAM; India region |

## 2. Data minimization (the core principle)
- **We never store raw face images.** The pipeline produces a 512-d embedding and discards
  the pixels. Embeddings are non-reversible-enough templates, dramatically lower risk than
  photos.
- The probe frame lives only in memory for the duration of one verification.

## 3. Encryption & key management
- **At rest**: `op-sqlite` compiled with **SQLCipher** (AES-256) for templates + attendance.
  `react-native-mmkv` with an encryption key for the small sync-state store.
- **Keys**: the DB key is generated once with a CSPRNG and stored in **Android Keystore
  (TEE/StrongBox)** / **iOS Keychain (Secure Enclave-backed)** via `react-native-keychain`,
  with `accessControl: BIOMETRY_ANY_OR_DEVICE_PASSCODE` and
  `WHEN_UNLOCKED_THIS_DEVICE_ONLY` (never backed up or synced to iCloud). The key is never
  in JS, AsyncStorage, or source.

## 4. Sync-then-purge lifecycle
1. Verify offline → write attendance row locally (`sync_state = pending`).
2. On `isInternetReachable`, batch-POST to the backend (idempotent by client UUID).
3. On confirmed `ok` per record → **delete** the synced rows, then `VACUUM` so freed
   SQLCipher pages are overwritten.
4. Enrollment templates are retained per HR/retention policy and are erasable on
   exit/request (right to erasure).

## 5. DPDP Act 2023 alignment
- **Consent & purpose limitation**, face data collected solely for attendance, with notice;
  no secondary use.
- **Storage limitation**, sync-then-purge + S3 lifecycle expiry erase data once its purpose
  is served.
- **Reasonable security safeguards**, encryption at rest/in transit, hardware-backed keys,
  least-privilege IAM.
- **Data Fiduciary duties**, NHAI is the Fiduciary; breach-notification readiness and
  per-person erasure are supported by the data model.
- **Localization**, all cloud resources in an **India AWS region (ap-south-1)**.

## 6. What leaves the device
| Data | Leaves device? | Notes |
|---|---|---|
| Raw face image | **No** | Never persisted or transmitted |
| Face embedding | Only as enrollment metadata if central enroll is enabled | Encrypted in transit |
| Attendance event (id, personId, ts, site, score) | Yes, on sync | Then purged locally |
| Device key / DB key | **No** | Stays in secure hardware |
