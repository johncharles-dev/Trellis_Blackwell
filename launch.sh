#!/bin/bash
# TRELLIS Launcher - Switch between Image-to-3D and Text-to-3D
# Usage: ./launch.sh [image|text]

CONDA_ENV="trellis-bw"
CONDA_BIN="/home/cj/miniconda3/bin/conda"
ENV_VARS="XFORMERS_DISABLED=1 ATTN_BACKEND=sdpa"
COMMON_ARGS="--precision auto --host 127.0.0.1"

# Kill any running TRELLIS instance
kill_existing() {
    pids=$(pgrep -f 'python.*app.*\.py.*--precision' 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "Stopping running TRELLIS server..."
        kill $pids 2>/dev/null
        sleep 2
    fi
}

case "${1:-}" in
    image|img|i)
        kill_existing
        echo "Starting Image-to-3D on http://127.0.0.1:7860"
        $CONDA_BIN run -n $CONDA_ENV bash -c "$ENV_VARS python -u app.py $COMMON_ARGS"
        ;;
    text|txt|t)
        kill_existing
        echo "Starting Text-to-3D on http://127.0.0.1:7861"
        $CONDA_BIN run -n $CONDA_ENV bash -c "$ENV_VARS python -u app_text.py $COMMON_ARGS"
        ;;
    stop|kill|k)
        kill_existing
        echo "All TRELLIS servers stopped."
        ;;
    *)
        echo "TRELLIS Launcher"
        echo "  ./launch.sh image   - Start Image-to-3D (port 7860)"
        echo "  ./launch.sh text    - Start Text-to-3D (port 7861)"
        echo "  ./launch.sh stop    - Stop all servers"
        ;;
esac
