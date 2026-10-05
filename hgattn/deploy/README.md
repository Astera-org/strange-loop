# Tools for working with the Kubernetes cluster

These are tools for non-interactive use of the Kubernetes cluster - for flexibly
launching individual jobs.

## Workflow

### Syncing code updates to the PVC

Any time code in `strange-loop`, `streamvis`, or `att3ntion` is updated, you need to
sync the updates to the PVC for the changes to be accessible when running on the
cluster.  The `sync-receiver` Job is invoked by `sync_code.sh` to simply untar a
stream onto the PVC

```bash
./deploy/sync_code.sh
```

### Building att3ntion and streamvis wheels on the cluster

If changes to att3ntion or streamvis were synced in step 1, new wheels need to be
built.  The `cache-wheels` Job builds the att3ntion and streamvis wheels in the
strange-loop repo virtual environment, and places them on hrbigelow-home.  They can
then be pip installed from strange-loop as dependencies.

```bash
kubectl apply -f ~/ai/projects/strange-loop/hgattn/deploy/cache-wheels.yaml
```

### Launch a training run using `hgattn.expt.train_general`

To launch a training run using train_general, do:

```bash
python -m hgattn.deploy.launch_job <args-for-train_general>
```

This python script applies `run-strange-loop.yaml` which defines the `strange-loop-*`
Jobs (generated names).  The job first creates an initContainer which sets up the
virtual environment for strange-loop, as well as installing the att3ntion and
streamvis wheels.  

## Other useful commands

```bash
# Installing autocomplete
source <(kubectl completions bash)

# Show jobs
kubectl get jobs
kubectl get job strange-loop<TAB> (shows completions)
kubectl get job strange-loop-28ggc -o json | jq ... 
kubectl get job strange-loop-28ggc -o yaml | yq ... 
kubectl get jobs -o yaml | yq -r \
    '.items[] | [.metadata.name] + (.spec.template.spec.containers[0].args // []) | join("\t")' \
    | column -t

# Diagnose a job
kubectl describe job <job-name>

# Diagnose a pod
kubectl describe pod -l job-name=<job-name>

# Follow logs during a running job
kubectl logs -l job-name=<job-name>
kubectl logs -l job-name=<job-name> -f # for live following
```
