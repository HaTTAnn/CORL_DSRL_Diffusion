#!/usr/bin/env bash
# MuJoCo 2.1.0 runtime for mujoco_py (required by robomimic/robosuite in this project).
# Usage: bash scripts/setup_mujoco210.sh
set -euo pipefail

MJOCO_URL="${MUJOCO210_URL:-https://ghfast.top/https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz}"
DEST="${HOME}/.mujoco"
mkdir -p "${DEST}"

if [[ ! -d "${DEST}/mujoco210/bin" ]]; then
	echo "Downloading MuJoCo 2.1.0 (mujoco210)..."
	tmp="$(mktemp /tmp/mujoco210.XXXXXX.tgz)"
	trap 'rm -f "${tmp}"' EXIT
	curl -fsSL -o "${tmp}" "${MJOCO_URL}"
	tar -xzf "${tmp}" -C "${DEST}"
	echo "Installed to ${DEST}/mujoco210"
else
	echo "Found existing ${DEST}/mujoco210"
fi

# mujoco_py uses OSMesa CPU build if no NVIDIA lib dir is found; empty /usr/lib/nvidia
# selects the EGL GPU shim when nvidia-smi exists (matches typical headless + GPU servers).
if [[ ! -d /usr/lib/nvidia ]]; then
	echo "Creating /usr/lib/nvidia (may require sudo on your machine)..."
	mkdir -p /usr/lib/nvidia 2>/dev/null || sudo mkdir -p /usr/lib/nvidia
fi

echo ""
echo "If mujoco_py fails to compile eglshim (missing GL/glew.h), install:"
echo "  Debian/Ubuntu: sudo apt-get install -y libglew-dev"
echo "  RHEL/CentOS:   sudo yum install -y glew-devel (or equivalent)"
echo ""
echo "Then add to your shell profile:"
echo "  export LD_LIBRARY_PATH=\"\${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia:\${LD_LIBRARY_PATH:-}\""
echo "  export MUJOCO_GL=egl"
echo ""
echo "(train_dsrl.py also prepends these paths when ~/.mujoco/mujoco210 exists.)"
