# YouTube Data API v3 upload workflow

This documents the reusable form of a verified YouTube Data API v3 resumable
upload of a finished 4K H.264/AAC MP4. Account-specific identifiers, title,
video ID, and exact activity details are intentionally omitted from this shared
skill. The generic client defaults to private and verifies upload and processing
status independently; a verified upload is not proof transcoding has finished.

## Decide who can execute

The uploader deliberately has no Google credentials or built-in access to a
personal credential store. `HelperTokenProvider` invokes an operator-owned,
private executable by absolute path, with `--account EMAIL` and, after a 401,
`--force-refresh`. The helper returns a JSON object containing only a short-lived
`access_token` on stdout. The parent captures stdout privately, keeps the token
in memory, and never writes it to upload state or logs. Helper stderr and failures
are suppressed. Keep the helper outside the shared repository and workspace.

Only the account holder or an explicitly authorized credential operator may
provide this helper and perform uploads. Other agents must hand execution to that
operator; installing this skill does not grant account access. Never put OAuth
codes, redirect URLs, refresh/access tokens, client secrets, keyring passwords,
or other credential values in repository files, logs, command-line arguments,
shell history, or public messages. Do not include the helper or credential
configuration in this shared skill.

At execution time, check the installed `gog --version` and `gog youtube --help`
for a supported upload command. Tool capabilities change; use the API client in
this skill only when live help confirms no supported native upload command.

Before creating or resuming a session, the uploader calls authenticated
`channels.list?part=id&mine=true` and refuses to proceed unless the expected
channel ID belongs to the token's account. An API read failure also stops the
upload. A separate read-only `gog youtube channels list --mine --account EMAIL
--json` can be used to choose/confirm the expected ID before execution. The
helper therefore needs an existing read scope that allows `channels.list`; the
uploader fails closed if the check is unavailable. Do not broaden scopes without
explicit authorization.

YouTube Data API v3 must be enabled in the OAuth client's Google Cloud project.
A 403 `accessNotConfigured` means enable the API in that project; do not retry
blindly or switch projects/credentials. If an account lacks the `youtube.upload`
scope, obtain authorization from its holder first. Preserve existing OAuth
scopes and add `https://www.googleapis.com/auth/youtube.upload` only through an
authorized consent flow. Read-only YouTube scope alone cannot upload.

## Run the generic client

Preflight the approved video with the project's normal media checks (container,
codec, dimensions, frame rate, duration, audio, and full decode). Do not transcode
or modify it in this upload skill. The script independently computes SHA-256,
size, path, device/inode, modification-time, and change-time identity before
creating or resuming an upload; it checks the open file identity before each chunk.

Example only; choose `--privacy` from the actual user instruction. Visibility
defaults to `private`; subscriber notifications default to `false`:

```sh
python3 /path/to/creative-skills/youtube-upload/scripts/youtube_upload.py \
  --source /absolute/path/to/approved-final.mp4 \
  --state "$HOME/.local/state/youtube-upload/project-final.json" \
  --account authorized@example.com \
  --token-helper /private/operator/path/youtube-access-token \
  --expected-channel CHANNEL_ID \
  --title 'Approved title' --category-id 10 \
  --privacy private
```

The helper path must be an absolute executable outside this repository. It must
accept `--account EMAIL` and optional `--force-refresh`, return JSON with one
`access_token` string, keep all credential material inside the authorized
credential boundary, and never log the token. It is intentionally not provided
or installed here: shared discovery must not grant all agents access to Gio's
personal account. The upload CLI never accepts credentials as options.

Use `--privacy unlisted` only when requested for that video; use `--privacy
public` only on explicit instruction. Do not use `--notify-subscribers` unless
requested. YouTube/API project audit restrictions may force `private`; the client
reads back actual privacy and reports a mismatch without changing it.

## Protocol and recovery guarantees

- Initiate one session with `POST https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status&notifySubscribers=false` (query value follows the selected notification option). JSON carries only title/category and requested visibility; headers pin source byte length and `video/mp4` type.
- Use the returned HTTPS Google resumable URL for aligned 16 MiB chunk `PUT`s with `Content-Range`; each chunk size is a multiple of 256 KiB except the final chunk. The state file is private (`0700` directory, `0600` file); an adjacent `0600` lock serializes concurrent runs for the same upload. State contains no OAuth token.
- Before sending data and after transport ambiguity, probe with an empty `PUT` and `Content-Range: bytes */TOTAL`. Honor HTTP 308 `Range`; retry transient/network errors a bounded number of times with backoff. On chunk 401, refresh once through the helper, then reconcile the session position before retrying data. Do not blindly repeat a chunk or create another session.
- A definite non-transient HTTP rejection is recorded. An ambiguous session-creation response is marked `initiation_uncertain` and refuses automatic resubmission because the server may have created a session without returning its URL. Reconcile the destination account/session manually before taking any new-session action; preserve the state file.
- Save the final video ID before verification and use it to make future invocations verify instead of upload again. Read `videos.list` fields `snippet,status,processingDetails,contentDetails`; require expected channel, title, and actual privacy. No privacy mutation is attempted.
- Report the watch URL, upload status, and YouTube processing status separately. An upload can be accepted while transcoding is still `processing`; do not claim availability/processing completion from the upload response alone.

The tests use mocked transport and fake in-memory tokens only. They cover metadata/default privacy, chunk alignment, resumable query/range behavior, 401 refresh and reconciliation, ambiguous POST no-retry, source/request mismatch, private state, channel/visibility verification, and duplicate avoidance. They never contact Google, read credentials, or upload media.
