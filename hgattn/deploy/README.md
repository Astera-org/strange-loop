# Tools for working with the Kubernetes cluster

## deploy/cache-wheels.yaml

The `cache-wheels` Job builds the att3ntion and streamvis wheels in the strange-loop
repo virtual environment, and places them on hrbigelow-home.  They can then be pip
installed from strange-loop as dependencies.

Useful commands:

```bash
# copy current state of code over to PVC.  This script internally uses the
sync-receiver.yaml Job
./deploy/sync_code.sh

# build the wheels
kubectl apply -f ~/ai/projects/strange-loop/hgattn/deploy/cache-wheels.yaml

# launch a run using `hgattn.expt.train_general` with custom arguments
python -m hgattn.deploy.launch_job <args-for-train_general>


# Show jobs
kubectl get jobs

# Diagnose a job
kubectl describe job <job-name>

# Diagnose a pod
kubectl describe pod -l job-name=<job-name>

# Follow logs during a running job
kubectl logs -l job-name=<job-name>
kubectl logs -l job-name=<job-name> -f # for live following
```



