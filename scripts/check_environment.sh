#!/usr/bin/env bash
set -euo pipefail
uname -a
python3 --version
for tool in ollama taskset chrt stress-ng; do
  if ! command -v "$tool"; then
    printf 'Missing: %s\n' "$tool"
  fi
done
printf '%s\n' 'Allowed controller CPUs:'
taskset -pc "$$"
printf '%s\n' 'Kernel RT runtime and period in microseconds:'
cat /proc/sys/kernel/sched_rt_runtime_us /proc/sys/kernel/sched_rt_period_us
printf '%s\n' 'CPU topology:'
lscpu -e=CPU,CORE,SOCKET,ONLINE
