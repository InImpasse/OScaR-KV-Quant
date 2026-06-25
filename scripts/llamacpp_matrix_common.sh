#!/usr/bin/env bash

llamacpp_preset_tokens() {
  case "$1" in
    short) printf '512\n' ;;
    medium) printf '2048\n' ;;
    long) printf '8192\n' ;;
    16k) printf '16384\n' ;;
    32k) printf '32768\n' ;;
    *)
      echo "Unsupported preset '$1'. Use one of: short,medium,long,16k,32k." >&2
      return 2
      ;;
  esac
}

llamacpp_cases_from_modes() {
  local modes="$1"
  local out=()
  local item
  IFS=',' read -r -a mode_list <<< "$modes"
  for item in "${mode_list[@]}"; do
    item="$(echo "$item" | xargs)"
    case "$item" in
      bf16|baseline_bf16) out+=(baseline_bf16) ;;
      int2|plain_int2) out+=(plain_int2) ;;
      oscar-int2|oscar_int2) out+=(oscar_int2) ;;
      int4|plain_int4) out+=(plain_int4) ;;
      oscar-int4|oscar_int4) out+=(oscar_int4) ;;
      *)
        echo "Unsupported mode '$item'. Use bf16,int2,oscar-int2,int4,oscar-int4." >&2
        return 2
        ;;
    esac
  done
  local IFS=,
  printf '%s\n' "${out[*]}"
}
