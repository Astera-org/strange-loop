import sys
import yaml
from kubernetes import client, config
from importlib import resources

def main():
    overrides = sys.argv[1:]

    config.load_kube_config()
    batch_v1 = client.BatchV1Api()

    path = resources.files() / "run-strange-loop.yaml"
    with open(path, "r") as fh:
        job_manifest = yaml.safe_load(fh)

    job_manifest["metadata"].setdefault("labels", {}).update(overrides)
    job_manifest["spec"]["template"]["metadata"].setdefault("labels", {}).update(overrides)

    app_container = job_manifest["spec"]["template"]["spec"]["containers"][0]
    app_container["args"] = overrides

    namespace = "strange-loop"
    job = batch_v1.create_namespaced_job(namespace=namespace, body=job_manifest)

    print(f"Launched Job: {job.metadata.name}")

if __name__ == "__main__":
    main()

