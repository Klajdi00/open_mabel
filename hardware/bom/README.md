# bom — bill of materials

| File | Contents |
|---|---|
| `MABEL_Build_Guide.pdf` | Parts list with vendor links, build options side by side, machined-part operations, volume pricing |
| `mabel_bom.pdf` | Costed BOM: method, currency handling, cost structure, build-or-buy, open items |
| `data/core.csv` | **Source of truth** — the fixed lines shared by every build (incl. filament) |
| `data/choices.csv` | Options for the interchangeable positions (compute, wrist/head/base cameras, LiDAR), tagged by build tier |
| `data/machined.csv` | Custom metal parts, operations, volume pricing |
| `data/taobao.csv` | Verified prices from Chinese vendors + search strings |
| `data/build_or_buy.csv` | Arms/hands vs. assembled alternatives |
| `data/open_items.csv` | Unresolved conflicts and uncosted hardware |
| `data/peer_platforms.csv` | Comparison platforms |
| `tools/build_bom.py` | The only thing that computes anything: CSV → `figures/`, `generated/` |

```bash
python3 tools/build_bom.py
```

The BOM splits into a **fixed core** (identical in every build) and **five choices**
(compute, wrist cameras, head camera, base depth camera, LiDAR). Three reference builds —
Essential, Recommended, Maximum — differ only in those choices. Totals for each are in
`mabel_bom.pdf`; read `data/open_items.csv` before quoting a total.

## Line status codes (`core.csv`)

| Status | Meaning |
|---|---|
| `sheet` | as purchased |
| `corrected` | part number fixed against the vendor listing |
| `disputed` | conflicting sources — see `open_items.csv` |
| `tbd` | not yet priced; carried at zero |
