# robosuite arenas

- **Source:** https://github.com/ARISE-Initiative/robosuite
  (`robosuite/models/assets/arenas/` + the 8 textures those arenas reference from
  `robosuite/models/assets/textures/`, master branch, fetched 2026-07-02).
- **License:** MIT — see `LICENSE` in this directory (Copyright (c) 2022 Stanford
  Vision and Learning Lab and UT Robot Perception and Learning Lab).
- **What was taken:**
  - `arenas/`: `empty_arena.xml`, `table_arena.xml`, `multi_table_arena.xml`,
    `bins_arena.xml`, `pegs_arena.xml`
  - `textures/`: brass-ambra, ceramic, dark-wood, light-gray-floor-tile,
    light-gray-plaster, light-wood, steel-brushed, steel-scratched (PNG)
  - The full 20 MB texture set and all robot/gripper/object assets were NOT taken.
- **Local modifications:** none. Arena XMLs keep their original relative texture
  paths (`../textures/*.png`), so they remain loadable in place.
