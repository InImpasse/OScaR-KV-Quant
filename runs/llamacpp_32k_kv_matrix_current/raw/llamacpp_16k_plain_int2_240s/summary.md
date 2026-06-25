| variant | status | KV | KV pool MiB | peak MiB | delta MiB | pp tok/s | tg tok/s | note |
|---|---|---|---:|---:|---:|---:|---:|---|
| plain_int2 | ok | q2_0/q2_0 | 240.0 | 3792 | 3673 | 180.0 | 44.1 |  |
| plain_int3 | unsupported |  |  |  |  |  |  | q3 KV cache is not exposed by this llama.cpp branch; common/arg.cpp supports f32,f16,bf16,q8_0,q4_0,q4_1,iq4_nl,q5_0,q5_1,q2_0. |
