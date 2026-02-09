#!/bin/bash
# TRELLIS Launcher - Switch between Image-to-3D and Text-to-3D
# Usage: ./launch.sh [image|text]

CONDA_ENV="trellis-bw"
PYTHON="/home/cj/miniconda3/envs/trellis-bw/bin/python"
COMMON_ARGS="--precision auto --host 127.0.0.1"
export XFORMERS_DISABLED=1
export ATTN_BACKEND=sdpa

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
        cd /home/cj/Trellis_Blackwell
        $PYTHON -u app.py $COMMON_ARGS
        ;;
    text|txt|t)
        kill_existing
        echo "Starting Text-to-3D on http://127.0.0.1:7861"
        cd /home/cj/Trellis_Blackwell
        $PYTHON -u app_text.py $COMMON_ARGS
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
