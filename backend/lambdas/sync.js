'use strict';
/**
 * NetraID attendance sync, AWS Lambda (Node 20, API Gateway HTTP API proxy).
 *
 * POST /v1/attendance/sync
 *   body: { records: [{ id, personId, ts, siteId, lat, lng, deviceId,
 *                       livenessPassed, matchScore }] }
 *   ->   { results: [{ id, status: 'ok' | 'rejected' }] }
 *
 * Idempotent: each record is upserted by its client UUID with a
 * ConditionExpression, so re-sends after a flaky ACK never duplicate. This is
 * what lets the device safely "sync then purge".
 */
const {
  DynamoDBClient,
} = require('@aws-sdk/client-dynamodb');
const {
  DynamoDBDocumentClient,
  PutCommand,
} = require('@aws-sdk/lib-dynamodb');

const ddb = DynamoDBDocumentClient.from(new DynamoDBClient({}));
const TABLE = process.env.ATTENDANCE_TABLE || 'NetraIDAttendance';

const json = (statusCode, body) => ({
  statusCode,
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

function validate(r) {
  return (
    r &&
    typeof r.id === 'string' &&
    typeof r.personId === 'string' &&
    Number.isFinite(r.ts) &&
    typeof r.deviceId === 'string'
  );
}

exports.handler = async (event) => {
  let payload;
  try {
    payload = JSON.parse(event.body || '{}');
  } catch {
    return json(400, { message: 'invalid JSON' });
  }
  const records = Array.isArray(payload.records) ? payload.records : null;
  if (!records) return json(400, { message: 'records[] required' });
  if (records.length > 200) return json(413, { message: 'batch too large' });

  // The device identity comes from the JWT authorizer claims (Cognito), not the body.
  const claims = event.requestContext?.authorizer?.jwt?.claims || {};
  const authedDevice = claims.sub || event.headers?.['x-device-id'];

  const results = await Promise.all(
    records.map(async (r) => {
      if (!validate(r)) return { id: r?.id ?? null, status: 'rejected' };
      try {
        await ddb.send(
          new PutCommand({
            TableName: TABLE,
            Item: {
              pk: `PERSON#${r.personId}`,
              sk: `TS#${r.ts}#${r.id}`,
              id: r.id,
              personId: r.personId,
              ts: r.ts,
              siteId: r.siteId ?? null,
              lat: r.lat ?? null,
              lng: r.lng ?? null,
              deviceId: r.deviceId,
              authedDevice: authedDevice ?? null,
              livenessPassed: !!r.livenessPassed,
              matchScore: r.matchScore ?? null,
              gsi1pk: r.siteId ? `SITE#${r.siteId}` : 'SITE#unknown',
              gsi1sk: `TS#${r.ts}`,
              createdAt: Date.now(),
            },
            // Idempotency: only write if this UUID was never seen.
            ConditionExpression: 'attribute_not_exists(id)',
          }),
        );
        return { id: r.id, status: 'ok' };
      } catch (e) {
        // Conditional failure == already stored == success for sync purposes.
        if (e.name === 'ConditionalCheckFailedException') {
          return { id: r.id, status: 'ok' };
        }
        console.error('put failed', r.id, e);
        return { id: r.id, status: 'rejected' };
      }
    }),
  );

  return json(200, { results });
};
