#!/usr/bin/env bash
# Shared helpers for repo-relative path defaults in shell scripts.
# Requires REPO_ROOT to be set before calling these functions.

resolve_repo_path() {
  local path="$1"
  if [[ "${path}" != /* ]]; then
    printf '%s/%s\n' "${REPO_ROOT}" "${path}"
  else
    printf '%s\n' "${path}"
  fi
}

repo_relative_path() {
  local path="$1"
  path="$(resolve_repo_path "${path}")"
  if [[ "${path}" == "${REPO_ROOT}/"* ]]; then
    printf '%s\n' "${path#${REPO_ROOT}/}"
  else
    printf '%s\n' "${path}"
  fi
}

setup_runtime_caches() {
  local cache_root="${CACHE_ROOT:-${REPO_ROOT}/.cache}"
  mkdir -p "${cache_root}"
  export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${cache_root}/flashinfer-base}"
  export FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-${cache_root}/flashinfer-cache}"
  export FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-${cache_root}/flashinfer-workspace}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${cache_root}/triton-cache}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${cache_root}/xdg-cache}"
  mkdir -p "${FLASHINFER_WORKSPACE_BASE}" "${FLASHINFER_CACHE_DIR}" "${FLASHINFER_WORKSPACE_DIR}" "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}"
}

setup_eval_output_dir() {
  local output_dir="${SGLANG_EVAL_OUTPUT_DIR:-${REPO_ROOT}/.cache/sglang-eval}"
  mkdir -p "${output_dir}"
  export SGLANG_EVAL_OUTPUT_DIR="${output_dir}"
}
