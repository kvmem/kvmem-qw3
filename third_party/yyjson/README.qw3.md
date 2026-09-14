Vendored yyjson 0.12.0 from https://github.com/ibireme/yyjson at commit
`8b4a38dc994a110abaec8a400615567bd996105f`. See LICENSE and provenance.json.

QW3 uses the strict reader for large native Anthropic request bodies. It enables
only YYJSON_READ_NUMBER_AS_RAW; the original nlohmann reader interprets numeric
literals and handles fallback/error cases. No JSON5, invalid Unicode, comments,
in-situ parsing or reduced validation options are enabled.

The library is linked only into the HTTP server and its parser test. It is not
part of the inference core or CUDA kernels. Tests/native_request_json_test.cpp
compares values, numeric types, serialization and errors with the old parser,
and accepts optional recorded request files for performance comparisons.
