# Heat-3D quick-start fixture

`heat3d_seed7_case07.pt` contains the archived +JCP model state and
zero-based test case 7 from the original seed-7 Heat-3D experiment.
It includes the clean initial field, the noisy observation with and without
training-set normalization, the solver and architecture settings, and the
SHA-256 hashes of the source files. Case 7 is closest to the median
state RMSE across the 80 test cases in a CPU replay.

| Original artifact | SHA-256 |
| --- | --- |
| `heat3d_dataset_seed7.pt` | `1d68d41f88e033fe481233f4b41b46b59ea70a07d910e8aad84d34ea33518848` |
| `heat3d_cnn_model_plus_jcp.pt` | `b3f950082d7f39f77ef6444cdc160f4c8a328a3e7aa97f47f86b1eb04ef32a72` |

The archived dataset's descriptive benchmark field says `allen_cahn2d`.
Its 512-component tensors, Heat-3D configuration, checksum, and replay
against the Heat-3D summary identify the actual contents.
