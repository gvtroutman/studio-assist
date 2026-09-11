# The container OpenCode runs in for Studio Assistant. Built once by the Start
# OpenCode button (studio_agent.ContainerSpec.launch) as the image named by
# OPENCODE_IMAGE, default "studio-opencode".
#
# What the container is given at run time - and all it is given - is decided
# by docker_run_args() in studio_agent.py: one bind mount (the workspace, at
# /workspace), one named volume for OpenCode's own state under /home/node, and
# port 4096 published to this PC's loopback. Nothing in here names a folder on
# the host.
FROM node:22-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates python3 ripgrep \
 && rm -rf /var/lib/apt/lists/*

RUN npm install -g opencode-ai@latest && npm cache clean --force

# The unprivileged user the node image ships with (uid 1000). Its home is the
# volume where OpenCode keeps sessions; the workspace is mounted over /workspace.
USER node
WORKDIR /workspace
ENV HOME=/home/node

EXPOSE 4096
ENTRYPOINT ["opencode", "serve", "--hostname", "0.0.0.0", "--port", "4096"]
