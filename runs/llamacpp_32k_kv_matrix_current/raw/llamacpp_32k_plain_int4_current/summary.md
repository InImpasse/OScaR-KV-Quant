| variant | status | KV | KV pool MiB | peak MiB | delta MiB | pp tok/s | tg tok/s | note |
|---|---|---|---:|---:|---:|---:|---:|---|
| plain_int4 | ok | q4_0/q4_0 | 720.0 | 4324 | 4205 | 2265.0 | 41.0 |  |
| plain_int3 | unsupported |  |  |  |  |  |  | q3 KV cache is not exposed by this llama.cpp branch; common/arg.cpp supports f32,f16,bf16,q8_0,q4_0,q4_1,iq4_nl,q5_0,q5_1,q2_0. |
