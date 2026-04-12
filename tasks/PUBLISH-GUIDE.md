# Publishing syara-x to crates.io

## Prerequisites

You need a crates.io account linked to your GitHub.

## Step 1: Log in to crates.io

1. Go to https://crates.io and click **Log in with GitHub**
2. Authorize the crates.io application if prompted

## Step 2: Generate an API token

1. Go to https://crates.io/settings/tokens
2. Click **New Token**
3. Name it something like `syara-x-publish`
4. Under scopes, select **publish-new** and **publish-update**
5. Click **Generate Token**
6. Copy the token (it will only be shown once)

## Step 3: Authenticate cargo locally

Run this in your terminal:

```bash
cargo login
```

Paste the API token when prompted. This stores the token in `~/.cargo/credentials.toml`.

**Security note:** The token is stored locally. Never commit `~/.cargo/credentials.toml` to git.

## Step 4: Commit all changes

Before publishing, all changes must be committed:

```bash
git add -A
git commit -m "Prepare for crates.io publish"
git push
```

## Step 5: Publish syara-x (main library first)

```bash
cargo publish -p syara-x
```

Wait for it to succeed. You can verify at https://crates.io/crates/syara-x

## Step 6: Publish syara-x-capi (depends on syara-x)

**Important:** Wait a minute or two after Step 5 for crates.io to index the new crate.

```bash
cargo publish -p syara-x-capi
```

Verify at https://crates.io/crates/syara-x-capi

## Step 7: Verify

Check both crate pages:
- https://crates.io/crates/syara-x
- https://crates.io/crates/syara-x-capi

Verify the README renders correctly and the metadata (license, repo link, keywords) is shown.

## Troubleshooting

**"crate name already taken"**: Someone else owns the name. You'd need to choose a different name or contact the owner.

**"no matching package named syara-x found" on capi publish**: The crates.io index hasn't updated yet. Wait 2-3 minutes and retry.

**Token expired or invalid**: Re-generate at https://crates.io/settings/tokens and re-run `cargo login`.

## Adding team ownership (optional)

To add co-owners to the crate:

```bash
cargo owner --add github:ORG_NAME:TEAM_NAME -p syara-x
# or for individual users:
cargo owner --add USERNAME -p syara-x
```
