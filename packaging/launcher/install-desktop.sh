#!/bin/sh
# Run after putting the portable folder in its final location. No sudo needed.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
applications=${XDG_DATA_HOME:-"$HOME/.local/share"}/applications
mkdir -p "$applications"
# Desktop Exec has its own quoting rules; no shell evaluates this command.
escaped=$(printf '%s' "$root" | sed 's/\\/\\\\\\\\/g; s/"/\\\\"/g; s/`/\\\\`/g; s/\$/\\\\$/g; s/%/%%/g')
icon=$(printf '%s' "$root" | sed 's/\\/\\\\/g')
case "$root" in
  *'
'*) printf '%s\n' 'Paths containing newlines cannot be used in desktop entries.' >&2; exit 1 ;;
esac
cat > "$applications/fox-reader.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Fox Reader
Comment=Manga reader with OCR and translation
Exec="$escaped/launcher"
Icon=$icon/icons/fox-reader.png
Terminal=false
Categories=Graphics;Viewer;
StartupWMClass=FoxReader
EOF
printf '%s\n' "Installed $applications/fox-reader.desktop"
