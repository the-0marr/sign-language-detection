# Dataset

Place the project dataset in a folder named `MP_Data` at the repository root.

The expected structure is:

```text
MP_Data/
├── action_1/
│   ├── 0/
│   │   ├── 0.npy
│   │   ├── 1.npy
│   │   └── ...
│   ├── 1/
│   └── ...
├── action_2/
└── ...
```

Each sequence contains 30 frames, and each frame contains 1,662 landmark features.

Do not commit personal recordings or large generated datasets to GitHub. Keep `MP_Data/` local unless a separate data-hosting strategy is used.
