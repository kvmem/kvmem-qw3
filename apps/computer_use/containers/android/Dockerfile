FROM happysixd/osworld-docker@sha256:0e6497a9295647cf05bf2b2af522fdd79bdeba2737595259cab310a3bcf6baa9
ARG QW3_UID=1000
RUN apt-get -o Acquire::http::Proxy=false -o Acquire::https::Proxy=false update \
    && apt-get -o Acquire::http::Proxy=false -o Acquire::https::Proxy=false install -y --no-install-recommends \
      libx11-6 libxcb1 libxau6 libxdmcp6 libxext6 libxrender1 libxi6 libxrandr2 \
      libnss3 libdbus-1-3 libglib2.0-0 libpulse0 libasound2t64 libegl1 libgl1 \
    && rm -rf /var/lib/apt/lists/*
# adb resolves its data home through passwd, including when HOME is unset.
RUN useradd --uid ${QW3_UID} --non-unique --home-dir /emulator-user --no-create-home --shell /bin/sh qw3
USER qw3
ENTRYPOINT ["/bin/sh", "/qw3-entrypoint.sh"]
