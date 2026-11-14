# Configuration

Fox Reader creates `config/fox_config.yaml` from built-in defaults on first
launch, creating the `config/` folder if necessary. Configuration files are
local user data: they are not tracked in the repository or included in build
archives.

When running from source, `config/` lives in the checkout root. In a packaged
installation, it lives beside the launcher, outside `bin/` and the Mac `.app`.
Keep this folder when moving or updating Fox Reader.

## Defaults and manual changes

The default configuration is:

```yaml
host: 127.0.0.1
port: 7954
mtl_dir: models
theme: dark
layout: basic
```

Close Fox Reader before editing a configuration file, then restart to apply
the changes. In particular, `host` and `port` cannot be changed through runtime
configuration updates.

| Field | Default | Meaning |
|---|---|---|
| `host` | `127.0.0.1` | Valid IP address or hostname on which the backend listens. Use the hostname alone, without a protocol or port. Internationalized names are normalized to their IDNA form. |
| `port` | `7954` | Integer from 1 to 65535. A quoted string of decimal digits is also accepted; booleans and floating-point numbers are invalid. |
| `mtl_dir` | `models` | Nonempty local translation-model path without control characters. Relative paths are resolved from the Fox Reader data root; absolute paths are also supported. |
| `theme` | `dark` | `dark` or `light`. |
| `layout` | `basic` | `basic` or `vulpes`. Older `default`, `reader`, and `translation` values remain supported. |

The retired `default` layout opens Vulpes; `reader` and `translation` open
Basic. Theme and layout names accept surrounding whitespace and differences
in letter case. For Windows paths, use forward slashes or YAML single quotes,
for example `mtl_dir: 'D:\Fox Models'`.

Host validation checks syntax locally; it does not look up DNS or check that
the address belongs to an available network interface. Model-directory
validation does not require the path to exist already. A valid saved preference
is therefore retained when a network interface or storage location is temporarily
unavailable.

This file controls only these application fields. Devices, API tokens, OCR
preferences, and per-language MTL choices live in `config/settings.yaml`.
Custom endpoints live in `config/user_endpoints.yaml`; folder/workspace state
lives in `config/workspaces.yaml`. Their settings remain separate.

## Optional `user_config.yaml`

You can manually create `config/user_config.yaml` using the same fields. Fox
Reader never creates this optional file automatically. When it exists, it is
the active configuration and receives changes saved through the app.

The file takes precedence as a whole. Missing fields use the built-in defaults,
not values from `fox_config.yaml`. For example, with this user file:

```yaml
theme: light
```

Fox Reader uses the light theme and the default port `7954`, even if
`fox_config.yaml` contains another port. The first launch fills the user file's
missing fields. Removing `user_config.yaml` while the app is closed makes the
next launch use `fox_config.yaml` again.

Fox Reader still creates and checks `fox_config.yaml` when a user file is
present, so both files are ready for future launches.

## Automatic updates and repairs

Each launch checks both the default file and any existing user file against
the current built-in configuration:

- Missing fields are added with their defaults. This also brings older files
  up to date when a Fox Reader update introduces new fields.
- Invalid known fields are replaced individually with their defaults. A bad
  port, for example, does not reset a valid theme or model directory.
- Unknown keys and their values remain on disk when the YAML parses
  successfully. The current version ignores them, preserving data belonging
  to another version or your own tooling.
- Duplicate keys are logged and reduced to their last value.
- Complete files already in the standard format are left byte-for-byte
  unchanged. A routine launch does not rewrite them.

To keep desktop startup consistent, Fox Reader writes `host` and `port` as
ordinary top-level YAML lines when necessary. Valid alternative representations,
such as flow-style mappings or quoted keys, are normalized once. An empty file
is filled with defaults.

If the YAML is broken, the document is not a mapping, or its encoding cannot
be decoded, Fox Reader replaces the document directly with defaults. Its old
fields, including unknown fields, cannot be preserved because the document
cannot be parsed reliably.
UTF-8 files with or without a byte-order mark and UTF-16 files are supported;
rewritten files use UTF-8.

Repairs update the configuration file directly. Fox Reader does not create
automatic configuration backups.

Writes replace the destination only after the new document has been written
successfully. If creation or repair cannot be saved, the configuration loader
uses the defaults or repaired values in memory and logs the persistence
failure. Files that cannot be opened for reading are left untouched. Changes
made through the app report save failures and retain the previous configuration.
If a file has changed on disk since loading, saving refuses to overwrite those
edits; restart Fox Reader to reload it before saving again.

Rewriting a file may change its formatting and remove YAML comments. Valid
settings and unknown fields are preserved when the original YAML parses
successfully. Keep your own copy if you need a particular handwritten format.

## Diagnostics and upgrades

Check `logs/backend.log` for configuration creation, added fields, repairs,
and read/write failures. Repair messages identify the file and affected field
without printing its stored value.

Automatic binary updates leave `config/` and `models/` untouched and merge
fonts. The next launch reconciles your existing configuration automatically.
For a manual update into a new folder, copy your old `config/`, `fonts/`, and
`models/` folders before starting it. There is no bundled YAML template to
merge. See [Updating Fox Reader](updating.md) for the complete update process.
