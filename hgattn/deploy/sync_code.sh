#!/usr/bin/env bash
set -eo pipefail

echo "Ensuring sync-receiver pod is running..."
kubectl apply -f ~/.kube/templates/sync-receiver.yaml
kubectl wait --for=condition=Ready pod/sync-receiver --timeout=60s

echo "Streaming repositories to PVC..."
tar -czf - \
	--exclude='.git' \
	--exclude='__pycache__' \
	--exclude='*.pyc' \
	--exclude='.venv' \
	--exclude='.mypy_cache' \
	--exclude='node_modules' \
	--exclude='.devspace' \
	-C "$HOME/ai/projects" strange-loop att3ntion streamvis | \
	kubectl attach sync-receiver -i --quiet

echo "Deleting sync-receiver pod"
kubectl delete pod sync-receiver

