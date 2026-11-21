# Job images

Images the `hgattn/deploy` Jobs run. CI builds them
(`.github/workflows/images.yml`) and pushes them to the private
`strange-loop` project on `harbor.astera.sh`; nobody pushes by hand.

| Directory | Image | Contents |
|---|---|---|
| `cuda` | `harbor.astera.sh/strange-loop/cuda:13.3.1-devel-ubuntu26.04-custom` | `nvidia/cuda:13.3.1-devel-ubuntu26.04` with build tools, git and `uv` |
| `mini-rsync` | `harbor.astera.sh/strange-loop/mini-rsync:latest` | Alpine with an rsync daemon serving `/workspace` |

## Changing an image

Open a pull request against `master` that edits its directory; CI checks
that both images build. Merging publishes both, each as `sha-<commit>`, and
moves the tag in the table to the new build. A CUDA version bump changes the
`FROM` line in `cuda/Dockerfile` and the `tag` in the workflow together.

To add an image, add a directory with a `Dockerfile` and list it in both
matrices of the workflow.

## Pulling

Pods in the `strange-loop` namespace pull with the `harbor-strange-loop-pull`
Secret, which the namespace's `default` ServiceAccount carries. A Pod that
sets its own `imagePullSecrets` must list that Secret too.

Harbor keeps, per image, the 30 newest `sha-*` tags plus any pulled in the
last 30 days, and every tag that is not `sha-*`.
