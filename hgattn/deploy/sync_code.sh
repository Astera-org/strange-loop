#!/usr/bin/env bash
set -eo pipefail
curdir=$(dirname $0)

echo "Ensuring sync-receiver pod is running..."
kubectl delete job sync-receiver --ignore-not-found
kubectl apply -f ${curdir}/sync-receiver.yaml
kubectl wait --for=condition=Ready pod -l job-name=sync-receiver --timeout=60s

echo "Streaming repositories to PVC..."
tar -czf - \
	--exclude='.git' \
	--exclude='__pycache__' \
	--exclude='*.pyc' \
	--exclude='.venv' \
	--exclude='.mypy_cache' \
	--exclude='node_modules' \
	--exclude='.devspace' \
  --exclude='build' \
	-C "$HOME/ai/projects" strange-loop att3ntion streamvis | \
	kubectl attach job/sync-receiver -i --quiet

