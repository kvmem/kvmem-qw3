# Isolated host-side decoder for saved XCF files. No VM access or model tools.
FROM python@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea
LABEL com.qw3.computer-use.owner="codex-goal-20260930"
RUN apt-get -o Acquire::http::Proxy=false -o Acquire::https::Proxy=false update \
    && apt-get -o Acquire::http::Proxy=false -o Acquire::https::Proxy=false install -y --no-install-recommends imagemagick \
    && dpkg-query -W imagemagick '*magick*' > /image-grader-packages.txt \
    && rm -rf /var/lib/apt/lists/*
ENV MAGICK_THREAD_LIMIT=2 MAGICK_MEMORY_LIMIT=256MiB MAGICK_MAP_LIMIT=512MiB MAGICK_DISK_LIMIT=512MiB
WORKDIR /tmp
