# cobi_plot.py

Plot energy-resolved crystal orbital bond indices (COBI) from LOBSTER output,
averaged over user-defined groups of bonds, with the integrated value (ICOBI)
reported for each group.

Bond groups can be built automatically from `ICOBILIST.lobster` — by element
pair, by bond length, and by the local coordination of the atoms involved — so
there is no need to assemble lists of hundreds of bond labels by hand.

---

## Contents

- [Requirements](#requirements)
- [Input files](#input-files)
- [Quick start](#quick-start)
- [Building groups automatically](#building-groups-automatically)
- [Building groups manually](#building-groups-manually)
- [Averaging, summing, combining](#averaging-summing-combining)
- [Appearance](#appearance)
- [Energy reference](#energy-reference)
- [Troubleshooting](#troubleshooting)
- [Option reference](#option-reference)

---

## Requirements

```
python >= 3.8
pymatgen
numpy
matplotlib
palettable
```

```bash
pip install pymatgen numpy matplotlib palettable
chmod +x cobi_plot.py
```

Tested with LOBSTER 5.1.1 and VASP 6.x.

---

## Input files

Run from the directory holding your LOBSTER output.

| File | Purpose |
|------|---------|
| `COBICAR.lobster` | Energy-resolved COBI curves |
| `ICOBILIST.lobster` | Integrated COBI, bond labels, atom pairs, distances |
| `POSCAR` | Structure: site ordering, used by `--classify` |

To produce the COBI files your `lobsterin` needs a generator line, e.g.

```
cohpGenerator from 0.1 to 6.0 orbitalwise
```

Every run writes `<output>.png` and `<output>.command`, the latter containing the
full command line for reproducibility.

---

## Quick start

```bash
cobi_plot.py --pairs 'Na-H;Be-H' --distmax 2.6 \
             --combine 'Total=Na-H+Be-H' \
             --colors 'Total=black' --dashed Total --show-icobi \
             --ymin -6 --ymax 8 --fontsize 14 --linewidth 2 \
             --output cobi_NaBeH3
```

This averages every Na–H and every Be–H bond shorter than 2.6 Å, adds a black
dashed curve for their sum, and prints the integrated COBI in the legend.

---

## Building groups automatically

The recommended route. The script reads the atom columns of
`ICOBILIST.lobster`, so it knows which elements and which sites each bond
connects.

```bash
--pairs 'Na-H;Be-H' --distmax 2.6
```

Pairs are order-insensitive: `Na-H` and `H-Na` are the same.

### The distance filter is effectively mandatory

A `cohpGenerator` running to 6 Å includes many non-bonded pairs; without a cutoff
they end up in your averages and drag the ICOBI down. The script warns when
`--distmax` is missing and reports the distance range of each group so you can
check where the cutoff falls:

```
  Na-H                    168 bonds   d = 2.287-2.489 A   mean ICOBI = 0.041
  Be-H                     72 bonds   d = 1.352-1.438 A   mean ICOBI = 0.324
```

If two bond types differ widely in length — Be–H near 1.4 Å, Na–H near 2.4 Å — a
single cutoff may not suit both. Inspect the printed ranges, and run twice if
needed. `--distmin` is available for the same purpose.

### Splitting by local environment

Groups can be subdivided according to the coordination of one atom in the bond:

```bash
--pairs 'Na-H;Be-H' --classify 'Be:1.8:H' --distmax 2.6
```

reads as *group the H sites by how many Be lie within 1.8 Å*, then split every
group containing H accordingly. Syntax: `NEIGHBOUR:CUTOFF[:TARGET]`, `TARGET`
defaulting to `H`. Periodic boundary conditions are respected.

Result:

```
  H classification: 6 sites at 1 Be, 6 sites at 2 Be
  Na-H [2 Be]              96 bonds   d = 2.312-2.489 A   mean ICOBI = 0.041
  Na-H [1 Be]              72 bonds   d = 2.287-2.401 A   mean ICOBI = 0.038
  Be-H [2 Be]              48 bonds   d = 1.421-1.438 A   mean ICOBI = 0.287
  Be-H [1 Be]              24 bonds   d = 1.352-1.359 A   mean ICOBI = 0.412
```

Comparing a terminal bond (one neighbour) against a bridging one (two) is a
direct probe of two-centre versus delocalised three-centre bonding.

The classification uses a single distance cutoff and no geometric criterion. If
the result disagrees with a coordination analysis such as CrystalNN, adjust the
cutoff — and check the two agree before publishing figures built on either.

---

## Building groups manually

The original interface still works and is mutually exclusive with `--pairs`:

```bash
--bonds 2 --labels '57,64,149,154;1017,1020,1082' --names 'Na-H;Be-H'
```

Labels are the bond numbers from `ICOBILIST.lobster`, groups separated by `;`.
Inclusive ranges written `15-20` are expanded. `--bonds` must match the number
of groups.

When migrating an existing manual command to `--pairs`, compare the bond counts
printed for each group against the number of labels you had listed. A manual
selection may encode a finer criterion than a plain distance cutoff, in which
case the averages will not match.

---

## Averaging, summing, combining

By default each group is **averaged** over its bonds, so groups of different
size stay comparable.

### `--sum`

```bash
--sum 'Total'
```

Switches the named groups to a raw sum (`divisor = 1`). Accepts group names or
1-based indices, comma-separated. Be aware that a summed group over hundreds of
bonds is orders of magnitude above the averaged ones, which flattens them on a
shared axis.

### `--combine`

```bash
--combine 'Total=Na-H+Be-H'
```

Adds a curve that is the sum of curves already plotted — the sum of two
*averages*, not an average over all bonds pooled together. This is usually what
you want for a "total": pooling would let the more numerous bond type dominate.

Several definitions chain with `;`, and a combined group can itself feed a later
definition. Group names must match exactly, brackets included, e.g.
`'Be-H total=Be-H [2 Be]+Be-H [1 Be]'`.

Note what the resulting curve represents: the sum of per-bond averages describes
a unit made of one bond of each type, not the whole cell. Say so in the caption
if you label it "Total".

---

## Appearance

```bash
--colors 'Total=black' --dashed Total --show-icobi
```

`--colors` takes `Name=colour` pairs separated by `;`. `--dashed` takes a
comma-separated list. Both accept group names or 1-based indices; indices are
easier when a name contains brackets, as `Be-H [2 Be]` does. Unlisted groups keep
the Set1 palette and a solid line.

`--show-icobi` appends the integrated value to each legend entry, formatted by
`--icobi-fmt` (default `.3f`, e.g. `.2f` for a lighter legend at small font
sizes). The value follows the group's own treatment:

| Group type | Legend value |
|------------|--------------|
| averaged (default) | mean ICOBI per bond |
| `--sum` | cumulative ICOBI |
| `--combine` | sum of the component means |

All three are consistent with the curve drawn beside them, but keep the
convention uniform if you tabulate these numbers across several compounds.

For a spin-polarised calculation, spin-down is drawn dashed; a group already set
to dashed uses a dotted line for spin-down so the two remain distinguishable.

---

## Energy reference

`--eshift X` shifts the energy axis, `E -> E - X`. `--zero-line` sets where the
dashed horizontal reference line is drawn on the final scale, or hides it with
`none`.

To place a COBI panel on the same scale as a DOS panel, take the shift printed
by the companion `dos_plot.py`:

```
  VBM = -0.0501 eV, CBM = 5.2566 eV, gap = 5.3066 eV
  (to align cobi_plot.py: --eshift -0.0501)
```

and pass it straight through:

```bash
cobi_plot.py --pairs 'Na-H;Be-H' --distmax 2.6 --eshift -0.0501 \
             --ymin -6 --ymax 8 --output cobi_panel
```

`COBICAR.lobster` and `DOSCAR.lobster` come from the same run and share the same
energy reference, so the shift transfers without further correction.

---

## Troubleshooting

**A bond label is not found.** The script reports and skips unknown labels. They
must exist in `ICOBILIST.lobster`, which is regenerated whenever the projection
changes.

**ICOBI values look too low.** Check the distance range printed for each group.
Non-bonded pairs pulled in by a generous `cohpGenerator` radius drag the average
down; tighten `--distmax`.

**`basisfunctions` in `lobsterin` seem ignored.** If `projectionData.lobster`
exists and `loadProjectionFromFile` is set, LOBSTER reloads the stored
projection and never applies your current basis lines. Delete or rename the
file, or comment out the keyword, then rerun. Verify what was used:

```bash
grep -A20 -i "basis functions" lobsterout
```

This matters for COBI: a stale projection silently changes every ICOBI.

**Charge spilling above roughly 5 %.** ICOBI values remain usable as trends but
their absolute magnitudes become unreliable, which undermines comparisons
between compounds. Check, in order: that the basis set keyword was applied
(`basisSet pbeVaspFit2015` vs `Koga` — compare both), that the POTCAR valence
matches the requested basis (`grep -E "VRHFIN|ZVAL" POTCAR`), and that the SCF
converged. High pressure raises spilling on its own, because tabulated basis
functions are fitted to free atoms; comparing the same phase across pressures
separates a setup problem from a physical trend.

**Many k-points fail orthonormalisation.** Set `ISYM = -1` in the static VASP
run. If it persists with symmetry already off, the basis is incomplete — the
same cause as high spilling. Quantify it:

```python
from pymatgen.io.lobster import Bandoverlaps
bo = Bandoverlaps("bandOverlaps.lobster")
print(max(bo.max_deviation))
```

**Site indices look wrong in `--classify`.** They follow the POSCAR ordering,
which must be the one used for the LOBSTER run.

---

## Option reference

| Option | Default | Description |
|--------|---------|-------------|
| `--pairs` | – | Build groups automatically, e.g. `'Na-H;Be-H'` |
| `--classify` | – | `NEIGHBOUR:CUTOFF[:TARGET]`, subdivides groups |
| `--distmin`, `--distmax` | – | Bond-length filter in Å |
| `--poscar` | `POSCAR` | Structure used by `--classify` |
| `--bonds` | – | Number of manual groups (with `--labels`) |
| `--labels` | – | Bond labels, groups separated by `;` |
| `--names` | – | Group names, `;`-separated |
| `--sum` | – | Groups to sum instead of average |
| `--combine` | – | `'Total=Na-H+Be-H'`, `;`-separated |
| `--colors` | – | `Name=colour`, `;`-separated |
| `--dashed` | – | Groups drawn dashed, comma-separated |
| `--show-icobi` | off | Show ICOBI in the legend |
| `--icobi-fmt` | `.3f` | ICOBI number format |
| `--eshift` | `0.0` | Energy shift, `E -> E - X` |
| `--zero-line` | `0` | Reference line position, or `none` |
| `--ymin`, `--ymax` | – | Energy window in eV |
| `--fontsize` | `20` | Axes and legend font size |
| `--linewidth` | `3.0` | Curve width |
| `--output` | `testcobi` | Output basename |

---

## Citing

- R. Nelson, C. Ertural, J. George, V. L. Deringer, G. Hautier, R. Dronskowski,
  *J. Comput. Chem.* **2020**, 41, 1931.
- S. P. Ong et al., *Comput. Mater. Sci.* **2013**, 68, 314.
