#!/bin/bash
# N137: B70 lock for 0000:48:00.0 ONLY (locks/xpu-<pci>.lock + .owner, same convention as N130). usage: <tag> <cmd...>
set -euo pipefail
pci=0000:48:00.0; tag=$1; shift
node=$(readlink -f /dev/dri/by-path/pci-0000:48:00.0-render)
[ -c "$node" ] || { echo "REFUSE absent render node" >&2; exit 65; }
[ "$(basename "$(readlink -f /sys/class/drm/$(basename $node)/device)")" = "$pci" ] || { echo "REFUSE node/pci mismatch" >&2; exit 65; }
[ "$(basename "$(readlink -f /sys/bus/pci/devices/$pci/driver)")" = xe ] || { echo "REFUSE non-xe device" >&2; exit 65; }
key=${pci//[:.]/_}; dir=$HOME/freetoken-exl3/locks; mkdir -p "$dir"
exec 5>"$dir/xpu-$key.lock"; flock 5
printf '%s %s %s\n' "$tag" "$node" "$(date -Is)" > "$dir/xpu-$key.owner"
trap ': > "$dir/xpu-$key.owner"' EXIT
export N137_XPU_PCI=$pci N137_XPU_RENDER=$node
"$@"
