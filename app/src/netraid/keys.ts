// Hardware-backed key management. The SQLCipher DB key is generated once and
// stored in Android Keystore / iOS Keychain, biometric-gated, never in JS or
// source. See docs/SECURITY_PRIVACY.md.

import 'react-native-get-random-values';
import * as Keychain from 'react-native-keychain';

const KEY_SERVICE = 'org.nhai.netraid.dbkey';

export async function getOrCreateDbKey(): Promise<string> {
  const existing = await Keychain.getGenericPassword({ service: KEY_SERVICE });
  if (existing) return existing.password;

  const bytes = new Uint8Array(32);
  crypto.getRandomValues(bytes);
  const key = Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');

  await Keychain.setGenericPassword('netraid', key, {
    service: KEY_SERVICE,
    accessControl: Keychain.ACCESS_CONTROL.BIOMETRY_ANY_OR_DEVICE_PASSCODE,
    accessible: Keychain.ACCESSIBLE.WHEN_UNLOCKED_THIS_DEVICE_ONLY, // not backed up / synced
    securityLevel: Keychain.SECURITY_LEVEL.SECURE_HARDWARE, // TEE / StrongBox on Android
  });
  return key;
}

/** Stable, non-PII device identifier persisted in the keychain. */
export async function getDeviceId(): Promise<string> {
  const svc = 'org.nhai.netraid.deviceid';
  const existing = await Keychain.getGenericPassword({ service: svc });
  if (existing) return existing.password;
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  const id =
    'dev_' +
    Array.from(bytes)
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('');
  await Keychain.setGenericPassword('deviceid', id, { service: svc });
  return id;
}
