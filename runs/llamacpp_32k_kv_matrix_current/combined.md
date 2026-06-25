| variant | status | prompt | KV | KV MiB | peak MiB | pp tok/s | tg tok/s | peak saved vs BF16 | KV saved vs BF16 | note |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| oscar_int2 | ok | 16384 | q2_0/q2_0 | 240.0 | 3796 | 183.7 | 28.0 |  |  |  |
| plain_int2 | ok | 16384 | q2_0/q2_0 | 240.0 | 3792 | 180.0 | 44.1 |  |  |  |
| baseline_bf16 | ok | 32768 | bf16/bf16 | 2560.0 | 6160 | 2486.4 | 41.6 | 0.0 | 0.0 |  |
| oscar_turbo2_streamk | ok | 32768 | turbo2/turbo2 | 340.0 | 3777 | 3984.7 | 40.0 | 2383.0 | 2220.0 |  |
| turbo2_streamk | ok | 32768 | turbo2/turbo2 | 340.0 | 3771 | 4256.7 | 39.0 | 2389.0 | 2220.0 |  |
| oscar_int4 | ok | 32768 | q4_0/q4_0 | 720.0 | 4324 | 2533.8 | 39.2 | 1836.0 | 1840.0 |  |
| plain_int4 | ok | 32768 | q4_0/q4_0 | 720.0 | 4324 | 2265.0 | 41.0 | 1836.0 | 1840.0 |  |
| oscar_int2 | failed | 32768 | q2_0/q2_0 | 480.0 | 4036 |  |  | 2124.0 | 2080.0 | missing or invalid llama-bench JSON |
