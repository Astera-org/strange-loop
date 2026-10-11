#!/usr/bin/env bash
set -eo pipefail
curdir=$(dirname $0)

if [ "$#" -ne 2 ]; then
  echo "Usage: sync_code.sh <source_directory> <dest dir inside /workspace> >&2"
  echo "sync_code.sh ~/path/to/strange-loop code >&2"
  exit 1
fi

source_dir=$1
dest_dir=$2

echo "Ensuring sync-receiver pod is running..."
kubectl delete job rsync-receiver --ignore-not-found
kubectl apply -f ${curdir}/rsync-receiver.yaml
POD_NAME=$(kubectl get pods -l job-name=rsync-receiver -o jsonpath='{.items[0].metadata.name}')
echo "Target pod: $POD_NAME"
kubectl wait --for=condition=Ready pod -l job-name=rsync-receiver --timeout=60s

kubectl port-forward $POD_NAME 8873:873 &
PF_PID=$!

cleanup() {
  kill "$PF_PID"
  kubectl delete job rsync-receiver
}

trap cleanup EXIT 

sleep 1

echo "Rsyncing repositories to PVC..."

rsync -av \
  --no-o --no-g \
  --delete \
	--exclude='.git' \
  --exclude='outputs/' \
	--exclude='__pycache__' \
	--exclude='*.pyc' \
	--exclude='.venv' \
	--exclude='.mypy_cache' \
	--exclude='node_modules' \
	--exclude='.devspace' \
  --exclude='build' \
  ${source_dir} \
  rsync://localhost:8873/workspace/${dest_dir}

