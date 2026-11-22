# Data sources

All files in `raw/` are kept exactly as transcribed. Unit conversion, band splitting and
the pairing of bands with compositions happen in `src/jominy/data.py`.

## `raw/tomasic_2025_heats.csv`

- **Source:** N. Tomašić, *Designing the Chemical Composition of Steel with the Required
  Hardenability Using Computational Methods*, doctoral thesis, University of Rijeka,
  Faculty of Engineering, 2025, Tables 3.2 (composition) and 3.3 (hardness).
  <https://riteh.uniri.hr/wp-content/uploads/PhD_Tomasic_Eps.pdf>
- **Content:** five measured Jominy curves (42CrMo4, 17CrNi6-6, C45E, 20CrMo5, 16MnCr5)
  at 1.5–50 mm. Composition gives C, Mn, Si, Cr, Ni, Mo and Cu in wt%. The thesis states
  ASTM grain size 7.
- **Transcribed:** 2026-11-22, from the text layer of the PDF. Each row has 13 values.
- **Note:** the thesis' full data set (470 compositions) is not published. Only these
  example rows are.

## `raw/ovako_steelnavigator_2026-11-22.json`

- **Source:** Ovako AB, *Steel Navigator* material data sheets,
  `https://steelnavigator.ovako.com/steel-grades/<grade>/`, retrieved 2026-11-22.
- **Content:** hardenability band polygons (HRC vs distance) and the composition range of
  the matching variant. The bands were read from the chart data embedded in the page.
  Nothing was digitised from images.
- **Included:** 17 boron-free bands from 8 grades: five ISO 683-2/-3 standard bands and
  twelve Ovako variants.
- **Excluded:**
  - 30MnB5 (three bands): the Creusot-Loire model has no boron term.
  - 20MnCr5 ISO 683-3: identical limits to 20MnCrS5, which is kept.
  - 20MnCr5 variant 236F: its composition table could not be read.
- **Missing lower limits:** where a standard gives no lower limit (below 20 HRC), the
  chart repeats the upper value. The loader drops those points.
- **Missing composition limits:** where a range gives no lower limit for Si, Ni, Mo or
  Cu, or omits the element, the loader uses the lowest content of that element in the
  five measured heats (`data.residual_levels`).
