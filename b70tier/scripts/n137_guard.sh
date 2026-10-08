#!/bin/bash
# usage: n137_guard.sh "<since YYYY-mm-dd HH:MM:SS>" <label>    exit 0 clean, 2 TRIPPED
# Rule (user, 2026-10-08): corrected AER allowed; stop on >30 corrected events in 10 min on 48:00.0's path
# (40:03.1 / 46:00.0 / 47:01.0 / 48:00.0 / 49:00.x), or any hard stop: fatal-pattern count above boot baseline (0),
# uncorrectable AER, pciehp, xe GT reset/wedged/forcewake on 48:00.0, MCE, D-state (same pid in D for 10 s).
since="$1"; label="${2:-check}"
c=$(timeout 60 journalctl -k -b -g "Completion-Wait loop timed out|Link Down|Card not present|reboot is needed" --no-pager 2>/dev/null | grep -vc "^-- ")
k=$(timeout 60 journalctl -k --since "$since" --no-pager 2>/dev/null)
k10=$(timeout 60 journalctl -k --since "-10min" --no-pager 2>/dev/null)
PATHRE="0000:(48:00\.0|47:01\.0|46:00\.0|40:03\.1|49:00\.[0-9])"
dev=$(printf '%s\n' "$k" | grep -E "$PATHRE" | grep -iE "reset|wedg|forcewake|link down|card not present|device lost|gt.*(hang|timed out)" | head -3)
pci=$(printf '%s\n' "$k" | grep -iE "pciehp" | grep -v "AttnBtn" | head -3)
aer=$(printf '%s\n' "$k" | grep -iE "event severity: *(fatal|recoverable|uncorrected)|type: *(fatal|recoverable|uncorrected)|Uncorrected \((Non-)?Fatal\)|severity=Uncorrected|AER: Uncorrectable" | head -3)
mce=$(printf '%s\n' "$k" | grep -iE "mce: |machine check" | grep -viE "decoding enabled|threshold limit" | head -3)
cor=$(printf '%s\n' "$k10" | grep -E "device_id: $PATHRE|$PATHRE.*AER: Corrected" | wc -l)
d1=$(ps -eo stat=,pid=,comm= | awk '$1 ~ /^D/ {print $2":"$3}' | sort); sleep 5
d2=$(ps -eo stat=,pid=,comm= | awk '$1 ~ /^D/ {print $2":"$3}' | sort); sleep 5
d3=$(ps -eo stat=,pid=,comm= | awk '$1 ~ /^D/ {print $2":"$3}' | sort)
dst=$(comm -12 <(echo "$d1") <(echo "$d2") | comm -12 - <(echo "$d3") | grep -v '^$' | tr '\n' ' ')
echo "[$(date '+%F %T')] guard $label count=$c path_corrected_10min=$cor dev='$dev' pciehp='$pci' aer='$aer' mce='$mce' dstate10s='$dst'"
if [ "$c" != 0 ] || [ "$cor" -gt 30 ] || [ -n "$dev" ] || [ -n "$pci" ] || [ -n "$aer" ] || [ -n "$mce" ] || [ -n "$dst" ]; then echo "TRIPPED $label"; exit 2; fi
exit 0
