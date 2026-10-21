---
name: youtube-upload
description: Use when publishing or resuming a finished MP4 on YouTube, choosing private/unlisted/public visibility, or checking upload versus transcoding status.
---

# YouTube Upload

Use for final YouTube delivery after a video is approved. This skill uses the
YouTube Data API v3 resumable upload protocol; it does not generate or modify
media. Read [the operating workflow](references/upload-workflow.md) before
execution.

## Access boundary

- Only an explicitly authorized account holder/credential operator may perform
  OAuth or upload actions. A shared skill install does not grant access to
  Gio's personal Google account. If you do not already have approved credentials
  and the YouTube upload scope, route execution to Zeus/the personal credential
  holder; do not request, copy, or expose secrets.
- Never initiate consent, change OAuth scopes, enable cloud APIs, or broaden
  privacy without applicable authorization from the relevant account holder.
- Do not create a second upload to test this skill. Use mocked tests and the
  saved private state to resume an approved upload.

## Run and verify

1. Confirm the approved source file, destination channel ID, title, category,
   privacy, and notification choice. Default visibility is `private`; request
   `unlisted` or `public` only when explicitly specified. The uploader confirms
   that the authorized account owns the expected channel before creating or
   resuming any upload; it needs an existing read scope that allows `channels.list`.
2. On the authorized host, read the live `gog --version` and
   `gog youtube --help`. If no native video-upload command is available, use
   `scripts/youtube_upload.py` with the operator-owned private token helper; do
   not pass access/refresh tokens on command lines or put them in environment
   dumps, logs, the repository, or shared state.
3. Keep resumable state outside the repository and workspace in a user-owned
   `0700` directory, with the state file `0600`. The state and its `.lock` sibling
   contain coordination metadata; the state contains a resumable session URL
   until completion. The uploader fingerprints the source and binds resume state
   to account, channel, title, category, privacy, and notifications.
4. The script defaults to `private` and `notifySubscribers=false`. A run uses
   one session, reconciles server position before chunks and after ambiguous
   failures, and never silently creates a replacement session. An ambiguous
   session-start response requires account reconciliation before retrying.
5. Treat a returned video ID and `videos.list` read-back as upload verification,
   not proof transcoding is finished. Check the actual channel and privacy;
   report `processingStatus` separately. The script never calls `videos.update`
   or silently changes privacy.

Run focused tests without network or credentials:

```sh
python3 -m unittest discover -s youtube-upload/tests -v
```
