| item | status | note |
|---|---|---|
| overall_status | complete | current deliverable is oscar_int4; exact INT2 remains a separate incomplete research target |
| exact_int2_research_status | incomplete | requires valid 32k q2_0/q2_0 speed/memory before exact INT2 can be called complete |
| 32k_bf16_baseline | complete | pp=2486.447077, peak=6160 |
| 32k_int4_memory_and_speed | complete | oscar_int4:pp=2533.822348,peak_saved=1836.0,kv_saved=1840.0; plain_int4:pp=2265.018455,peak_saved=1836.0,kv_saved=1840.0 |
| int4_cli_quality_smoke | complete | baseline_bf16_gpqa=3/10; baseline_bf16_gsm8k=4/10; oscar_int4_gpqa=4/10; oscar_int4_gsm8k=4/10 |
| 16k_int2_gate | complete | oscar_pp=183.693112, plain_pp=179.997972 |
| 32k_turbo2_reference | complete | oscar_turbo2_status=ok,oscar_turbo2_pp=3984.711113,oscar_turbo2_peak=3777.0,note=Turbo2 is not exact OSCAR INT2 |
| 32k_int2_speed_target | incomplete | exact_q2_status=failed,exact_q2_kv=q2_0/q2_0,exact_q2_pp=None,exact_q2_peak=4036.0,oscar_int4_pp=2533.822348,oscar_int4_peak=4324.0,bf16_pp=2486.447077,bf16_peak=6160.0; turbo2_reference_pp=3984.711113 |
| cuda_graph_512_ab | complete | graph_on_pp_pct_vs_off=-0.90 |
| llamacpp_only_guardrails | complete | markers present |
| execution_safety_guardrails | complete | entrypoints=markers present; legacy=markers present |
| no_gpu_verifier_guard | complete | verifier=markers present; q2_profile=markers present |
| recovery_command_guardrails | complete | matrix=markers present; q2_ramp=markers present |
| post_case_cooldown_guard | complete | markers present |
| recovery_readiness_report | complete | markers present |
| q2_ramp_gate_guard | complete | markers present |
| q2_ramp_gate_harness_guard | complete | markers present |
| futuremls_q2_cuda_port_plan | complete | checker=markers present; doc=markers present |
| q2_cuda_static_guardrails | complete | markers present |
| q2_cuda_path_archive | complete | archived path facts present (6 rows) |
